"""
A datamanager for the NSROS Bridge.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Tuple

from rich.console import Console

from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

from nsros.ros_dataset import ROSDataset
from nsros.ros_dataloader import ROSDataloader
from nsros.ros_dataparser import ROSDataParserConfig

import torch


CONSOLE = Console(width=120)


@dataclass
class ROSDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A ROS datamanager that handles a streaming dataloader."""

    _target: Type = field(default_factory=lambda: ROSDataManager)
    dataparser: ROSDataParserConfig = ROSDataParserConfig()
    """ Must use only the ROSDataParser here """
    publish_training_posearray: bool = True
    """ Whether the dataloader should publish an pose array of the training image poses. """
    data_update_freq: float = 5.0
    """ Frequency, in Hz, that images are added to the training dataset tensor. """
    num_training_images: int = 500
    """ Number of images to train on (for dataset tensor pre-allocation). """


class ROSDataManager(
    base_datamanager.VanillaDataManager
):  # pylint: disable=abstract-method
    """Essentially the VannilaDataManager from Nerfstudio except that the
    typical dataloader for training images is replaced with one that streams
    image and pose data from ROS.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ROSDataManagerConfig
    train_dataset: ROSDataset
    eval_dataset: ROSDataset

    def create_train_dataset(self) -> ROSDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train", num_images=self.config.num_training_images
        )
        return ROSDataset(
            dataparser_outputs=self.train_dataparser_outputs, device=self.device
        )

    def setup_train(self):
        assert self.train_dataset is not None
        self.train_image_dataloader = ROSDataloader(
            self.train_dataset,
            self.config.publish_training_posearray,
            self.config.data_update_freq,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )

        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras,
            self.train_camera_optimizer,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        First, checks for updates to the ROSDataloader, and then returns the next
        batch of data from the train dataloader.
        """
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        if image_batch['image'].numel() == 0:
            self.train_dataset = torch.load("/home/eiyike/ASCCFINAL/nerf_bridge/dataset.pt")
            self.train_image_dataloader = ROSDataloader(
                self.train_dataset,
                self.config.publish_training_posearray,
                self.config.data_update_freq,
                device=self.device,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
                batch_size=1,
            )

            self.iter_train_image_dataloader = iter(self.train_image_dataloader)
            breakpoint()
            self.train_pixel_sampler = self._get_pixel_sampler(
                self.train_dataset, self.config.train_num_rays_per_batch
            )
            self.train_camera_optimizer = self.config.camera_optimizer.setup(
                num_cameras=self.train_dataset.cameras.size, device=self.device
            )
            self.train_ray_generator = RayGenerator(
                self.train_dataset.cameras,
                self.train_camera_optimizer,
            )

        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = ROSDataloader(
            self.eval_dataset,
            self.config.publish_training_posearray,
            self.config.data_update_freq,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(
            self.eval_dataset, self.config.eval_num_rays_per_batch
        )
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device
        )
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras,
            self.eval_camera_optimizer,
        )

    def create_eval_dataset(self):
        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="test", num_images=self.config.num_training_images
        )
        return ROSDataset(
            dataparser_outputs=self.eval_dataparser_outputs, device=self.device
        )

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataset:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")
