# cuda packages
apt-get -y update && apt-get -y --no-install-recommends install git curl neovim clang pkg-config libfontconfig-dev opencl-headers ocl-icd-opencl-dev nvidia-cuda-toolkit;
curl --proto '=https' --tlsv1.2 -ssf https://sh.rustup.rs | sh -s -- -y;
. "$HOME/.cargo/env";
cd $HOME
git clone https://github.com/SamuelGallay/condensate.git;
cd condensate;
cargo build --release;

# nvidia-cuda-toolkit
# scp yourusername@yourserver:/home/yourusername/examplefile .
# nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv
# scp -P 40627 -r root@83.7.208.186:/root/condensate/archive .
