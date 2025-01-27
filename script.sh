# cuda packages
apt-get -y update && apt-get -y --no-install-recommends install git curl neovim clang pkg-config libfontconfig-dev ffmpeg opencl-headers ocl-icd-opencl-dev;
curl --proto '=https' --tlsv1.2 -ssf https://sh.rustup.rs | sh -s -- -y;
. "$HOME/.cargo/env";
git clone https://github.com/SamuelGallay/navier.git;
cd navier;
cargo build --release;
