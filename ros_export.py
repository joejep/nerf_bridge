import tyro
from typing import Union
from typing_extensions import Annotated, Literal
from nsros.ros_exporter import ExportPointCloud, ExportCameraPoses, ExportTSDFMesh, ExportPoissonMesh, ExportMarchingCubesMesh

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa