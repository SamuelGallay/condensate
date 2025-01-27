FROM rocm/dev-ubuntu-22.04 as build
WORKDIR /navier
LABEL maintainer="agagoo38@gmail.com" 
RUN  apt-get -y update && apt-get -y --no-install-recommends install git curl neovim clang opencl-headers pkg-config libfontconfig-dev ocl-icd-opencl-dev ffmpeg;
RUN curl --proto '=https' --tlsv1.2 -ssf https://sh.rustup.rs | sh -s -- -y;
COPY . /navier
RUN /root/.cargo/bin/cargo build --release;

FROM ubuntu:24.10 as main
RUN apt-get -y update && apt-get -y --no-install-recommends install ffmpeg mesa-opencl-icd 
ENV RUSTICL_ENABLE=radeonsi
COPY --from=build /navier/target/release/navier /bin/navier


