__kernel void diffusion(__global float2* phi,  __global float2* dx_phi,  __global float2* dy_phi, int N, float L, float dt) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float freqx = (2.0 * M_PI_F / L) * ((float)i - (float)N * (2*i >= N));
    float freqy = (2.0 * M_PI_F / L) * ((float)j - (float)N * (2*j >= N));
    float scalar = exp(-0.5 * (freqx*freqx + freqy*freqy) * dt);
    float phix = phi[i*N +j].x * scalar;
    float phiy = phi[i*N +j].y * scalar;
    
    phi[i*N+j].x = phix;
    phi[i*N+j].y = phiy;
    dx_phi[i*N+j].x = -phiy * freqx;
    dx_phi[i*N+j].y =  phix * freqx;
    dy_phi[i*N+j].x = -phiy * freqy;
    dy_phi[i*N+j].y =  phix * freqy;
}
__kernel void rotation(__global float2* phi, __global float2* dx_phi, __global float2* dy_phi, __global float2* phi2hat, int N, float L, float omega, float beta, float dt) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    float dx = L / N;
    float x = i*dx - L/2;
    float y = j*dx - L/2;
    float phix = phi[i*N +j].x; 
    float phiy = phi[i*N +j].y;

    phix += omega * (x*dy_phi[i*N+j].y - y*dx_phi[i*N+j].y) * dt;
    phiy -= omega * (x*dy_phi[i*N+j].x - y*dx_phi[i*N+j].x) * dt;
    
    float phi2 = phix*phix + phiy*phiy;
    float v2 = x*x + y*y;
    
    phix *= exp(-dt * (v2 + beta*phi2)); 
    phiy *= exp(-dt * (v2 + beta*phi2)); 

    phi[i*N+j].x = phix; 
    phi[i*N+j].y = phiy; 
    phi2hat[i*N+j].x = phix*phix + phiy*phiy; 
    phi2hat[i*N+j].y = 0; 
}

__kernel void rescale(__global float2* phi, __global float2* phi2hat, int N, float L) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float norm = sqrt(phi2hat[0].x) * L / N;
    phi[i*N+j].x /= norm;
    phi[i*N+j].y /= norm;
}
