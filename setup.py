import os
import re
import sys
import platform
import subprocess
from pathlib import Path
import site

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# taken from pybind cmake example
# https://github.com/pybind/cmake_example/blob/master/setup.py

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(
        self, name: str, sourcedir: str = "", extra_args: dict = None
    ) -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        if extra_args is not None:
            self.extra_args = extra_args
            self.debug = extra_args.get("--debug", False)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve()
        # only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(
            ext.name
        )  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = (
            int(os.environ.get("DEBUG", 0))
            if self.debug is None
            else self.debug
        )
        # overwrite if ext.debug exists
        debug = ext.debug if hasattr(ext, "debug") else debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        # MFEM args
        cmake_args += [
            "-DMFEM_USE_SUITESPARSE=ON",
            # "-DMFEM_USE_OPENMP=ON",
            # "-DMFEM_THREAD_SAFE=ON",
            # "-DMFEM_USE_LAPACK=ON",
            "-DMIMI_COMPILE_SPLINEPY=OFF",
            f"-DCMAKE_PREFIX_PATH={site.getsitepackages()[0]}",
        ]

        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item
            ]

        dup_prefix = []
        for i, ca in enumerate(cmake_args):
            if ca.startswith("-DCMAKE_PREFIX_PATH"):
                dup_prefix.append(i)
        if len(dup_prefix) > 1:
            prefix_path = []
            for dp in dup_prefix[::-1]:
                prefix_path.append(cmake_args.pop(dp))
            prefix_path = ";".join(prefix_path).replace(
                "-DCMAKE_PREFIX_PATH=", ""
            )
            prefix_path = "-DCMAKE_PREFIX_PATH=" + prefix_path
            cmake_args.append(prefix_path)

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        "-DCMAKE_MAKE_PROGRAM"
                        f":FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(
                x in cmake_generator for x in ("NMake", "Ninja")
            )

            # CMake allows an arch-in-generator style
            # for backward compatibility
            contains_arch = any(x in cmake_generator for x in ("ARM", "Win64"))

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_"
                    f"{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        serial = bool(int(os.environ.get("SERIAL", 0)))
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ and not serial:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call,
            # not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]
            else:
                # always full parallel
                build_args += [f"-j{os.cpu_count()}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True,
            env=os.environ,
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
        )
        print()  # Add an empty line for cleaner output


setup(
    name="mimi",
    version="0.0.0",
    author="Jaewook Lee",
    author_email="jaewooklee042@gmail.com",
    description="Collection of mfem based IGA computations.",
    long_description="",
    packages=["mimi"],
    ext_modules=[CMakeExtension("mimi.mimi_core")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
