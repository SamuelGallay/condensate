inline float mod(float2 a){
    return sqrt(a.x*a.x + a.y*a.y);
}

inline float mod2(float2 a){
    return a.x*a.x + a.y*a.y;
}

inline float2 conj(float2 a){
    return (float2) (a.x, -a.y);
}

inline float2 mul(float2 a, float2 b){
    return (float2) (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}


__kernel void mul_vect_scalar(__global float2* vect,  float2 scalar,  __global float2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = mul(vect[k], scalar);
}

__kernel void add_vect_scalar(__global float2* vect,  float2 scalar,  __global float2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = vect[k] + scalar;
}

__kernel void mul_vect_vect(__global float2* a,  __global float2* b,  __global float2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = mul(a[k], b[k]);
}

__kernel void add_vect_vect(__global float2* a,  __global float2* b,  __global float2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = a[k] + b[k];
}

__kernel void conj_vect(__global float2* a,  __global float2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = conj(a[k]);
}

__kernel void abs_squared_vect(__global float2* a, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    a[k] = (float2) (mod2(a[k]), 0.0);
}

__kernel void inv_sqrt_vect(__global float2* a, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    a[k] = (float2) (1.0/sqrt(mod2(a[k])), 0.0);
}

__kernel void scal(__global float2* a, __global float2* b, __global float2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = mul(a[k], conj(b[k]));
}



__kernel void energy(__global float2* phi,  __global float2* dxphi,  __global float2* dyphi, __global float2* out, float beta, float omega, float L, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    float dx = L / N;
    float x = i*dx - L/2;
    float y = j*dx - L/2;
    float v = x*x + y*y;

    float2 p = phi[k];
    float2 dxp = dxphi[k];
    float2 dyp = dyphi[k];

    float s = 0;
    s += 0.5f * (mod2(dxp) + mod2(dyp));
    s += v * mod2(p);
    s += beta/2 * mod2(p) * mod2(p);
    s += omega * mul(p, conj(x*dyp-y*dxp)).y;
    
    out[k] = (float2)(s, 0);
}

__kernel void alpha(__global float2* phi,  __global float2* dxphi,  __global float2* dyphi, __global float2* out, float beta, float L, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    float dx = L / N;
    float x = i*dx - L/2;
    float y = j*dx - L/2;
    float v = x*x + y*y;

    float2 p = phi[k];
    float2 dxp = dxphi[k];
    float2 dyp = dyphi[k];

    float s = 0;
    s += 0.5f * (mod2(dxp) + mod2(dyp));
    s += v * mod2(p);
    s += beta * mod2(p) * mod2(p);
    
    out[k] = (float2)(s, 0);
}

__kernel void hphif(__global float2* phi, __global float2* ff, __global float2* dxff,  __global float2* dyff, __global float2* lapff, __global float2* out, float beta, float omega, float L, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    float dx = L / N;
    float x = i*dx - L/2;
    float y = j*dx - L/2;
    float v = x*x + y*y;

    float2 p = phi[k];
    float2 f = ff[k];
    float2 dxf = dxff[k];
    float2 dyf = dyff[k];
    float2 lapf = lapff[k];

    float2 s = 0;
    s -= 0.5f * lapf;
    s += v * f;
    s += beta * mod2(p) * f;
    s += omega * mul((float2)(0, 1.0f), x*dyf - y*dxf);
    
    out[k] = s;
}

__kernel void differentiate(__global float2* phi,  __global float2* dx_phi,  __global float2* dy_phi, __global float2* lap_phi, float L, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float freqx = (2.0 * M_PI_F / L) * ((float)i - (float)N * (2*i >= N));
    float freqy = (2.0 * M_PI_F / L) * ((float)j - (float)N * (2*j >= N));
    float2 p = phi[i*N+j];
    
    dx_phi[i*N+j].x = -p.y * freqx;
    dx_phi[i*N+j].y =  p.x * freqx;
    dy_phi[i*N+j].x = -p.y * freqy;
    dy_phi[i*N+j].y =  p.x * freqy;
    lap_phi[i*N+j] = -(freqx*freqx + freqy*freqy) * p;
}

__kernel void invamlap2(__global float2* phi, float a, float L, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float freqx = (2.0 * M_PI_F / L) * ((float)i - (float)N * (2*i >= N));
    float freqy = (2.0 * M_PI_F / L) * ((float)j - (float)N * (2*j >= N));
    
    phi[i*N+j] /= a + 0.5f*(freqx*freqx + freqy*freqy);
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
