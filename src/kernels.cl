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

__kernel void rescale(__global float2* inphi, __global float2* outphi,  __global float2* diff, int N, float dx, float precomputed_norm, __global float2* sum_buffer) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float norm = precomputed_norm;
    //norm = sqrt(sum_buffer[0].x) * dx;
    
    float prevx = outphi[i*N+j].x;
    float prevy = outphi[i*N+j].y;

    float newx = inphi[i*N+j].x / norm;
    float newy = inphi[i*N+j].y / norm;

    outphi[i*N+j].x = newx;
    outphi[i*N+j].y = newy;

    diff[i*N+j].x = (newx-prevx)*(newx-prevx) + (newx-prevy)*(newy-prevy);
    diff[i*N+j].y = 0;
}

__kernel void sum(__global float2* in_buffer, ulong in_offset, __global float2* out_buffer,  ulong out_offset) {
    int i = get_global_id(0);
    out_buffer[out_offset + i] = in_buffer[in_offset + 2*i] + in_buffer[in_offset + 2*i+1];
}

__kernel void sum_inplace(__global float2* buffer, ulong out_size) {
    int i = get_global_id(0);
    buffer[i] += buffer[i + out_size] ;
}

__kernel void simd(__global float4* buffer, ulong out_size) {
    int i = get_global_id(0);
    float4 s = buffer[i];
    s+= buffer[i + out_size];
    s+= buffer[i + 2*out_size];
    s+= buffer[i + 3*out_size];
    s+= buffer[i + 4*out_size];
    s+= buffer[i + 5*out_size];
    s+= buffer[i + 6*out_size];
    s+= buffer[i + 7*out_size];
    s+= buffer[i + 8*out_size];
    s+= buffer[i + 9*out_size];
    s+= buffer[i + 10*out_size];
    s+= buffer[i + 11*out_size];
    s+= buffer[i + 12*out_size];
    s+= buffer[i + 13*out_size];
    s+= buffer[i + 14*out_size];
    s+= buffer[i + 15*out_size];
    buffer[i] = s;
}
