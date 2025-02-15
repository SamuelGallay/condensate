inline double mod(double2 a){
    return sqrt(a.x*a.x + a.y*a.y);
}

inline double mod2(double2 a){
    return a.x*a.x + a.y*a.y;
}

inline double2 conj(double2 a){
    return (double2) (a.x, -a.y);
}

inline double2 mul(double2 a, double2 b){
    return (double2) (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}


struct Params {
    ulong n;
    ulong niter;
    double length;
    double omega;
    double beta;
    double gamma;
    double dx;
};

__kernel void read_params(struct Params p){
    printf("This is my omega: %f\n", p.omega);
}

__kernel void mul_vect_scalar(__global double2* vect,  double2 scalar,  __global double2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = mul(vect[k], scalar);
}

__kernel void add_vect_scalar(__global double2* vect,  double2 scalar,  __global double2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = vect[k] + scalar;
}

__kernel void mul_vect_vect(__global double2* a,  __global double2* b,  __global double2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = mul(a[k], b[k]);
}

__kernel void add_vect_vect(__global double2* a,  __global double2* b,  __global double2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = a[k] + b[k];
}

__kernel void conj_vect(__global double2* a,  __global double2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = conj(a[k]);
}

__kernel void abs_squared_vect(__global double2* a, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    a[k] = (double2) (mod2(a[k]), 0.0);
}

__kernel void inv_sqrt_vect(__global double2* a, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    a[k] = (double2) (1.0/sqrt(mod2(a[k])), 0.0);
}

__kernel void scal(__global double2* a, __global double2* b, __global double2* out, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*N+j;
    out[k] = mul(a[k], conj(b[k]));
}



__kernel void energy(__global double2* phi,  __global double2* dxphi,  __global double2* dyphi, __global double2* out, struct Params params) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*params.n+j;
    double L = params.length;
    double dx = L / params.n;
    double x = i*dx - L/2;
    double y = j*dx - L/2;
    double v = x*x + y*y;

    double2 p = phi[k];
    double2 dxp = dxphi[k];
    double2 dyp = dyphi[k];

    double s = 0;
    s += 0.5f * (mod2(dxp) + mod2(dyp));
    s += v * mod2(p);
    s += params.beta/2 * mod2(p) * mod2(p);
    s += params.omega * mul(p, conj(x*dyp-y*dxp)).y;
    
    out[k] = (double2)(s, 0);
}

__kernel void alpha(__global double2* phi,  __global double2* dxphi,  __global double2* dyphi, __global double2* out, struct Params params) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*params.n+j;
    double L = params.length;
    double dx = L / params.n;
    double x = i*dx - L/2;
    double y = j*dx - L/2;
    double v = x*x + y*y;

    double2 p = phi[k];
    double2 dxp = dxphi[k];
    double2 dyp = dyphi[k];

    double s = 0;
    s += 0.5f * (mod2(dxp) + mod2(dyp));
    s += v * mod2(p);
    s += params.beta * mod2(p) * mod2(p);
    
    out[k] = (double2)(s, 0);
}

__kernel void hphif(__global double2* phi, __global double2* ff, __global double2* dxff,  __global double2* dyff, __global double2* lapff, __global double2* out, struct Params params) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = i*params.n+j;
    double L = params.length;
    double dx = L / params.n;
    double x = i*dx - L/2;
    double y = j*dx - L/2;
    double v = x*x + y*y;

    double2 p = phi[k];
    double2 f = ff[k];
    double2 dxf = dxff[k];
    double2 dyf = dyff[k];
    double2 lapf = lapff[k];

    double2 s = 0;
    s -= 0.5f * lapf;
    s += v * f;
    s += params.beta * mod2(p) * f;
    s += params.omega * mul((double2)(0, 1.0f), x*dyf - y*dxf);
    
    out[k] = s;
}

__kernel void differentiate(__global double2* phi,  __global double2* dx_phi,  __global double2* dy_phi, __global double2* lap_phi, struct Params params) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N = params.n;
    double L = params.length;
    double freqx = (2.0 * M_PI_F / L) * ((double)i - (double)N * (2*i >= N));
    double freqy = (2.0 * M_PI_F / L) * ((double)j - (double)N * (2*j >= N));
    double2 p = phi[i*N+j];
    
    dx_phi[i*N+j].x = -p.y * freqx;
    dx_phi[i*N+j].y =  p.x * freqx;
    dy_phi[i*N+j].x = -p.y * freqy;
    dy_phi[i*N+j].y =  p.x * freqy;
    lap_phi[i*N+j] = -(freqx*freqx + freqy*freqy) * p;
}

__kernel void invamlap2(__global double2* phi, double a, double L, ulong N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    double freqx = (2.0 * M_PI_F / L) * ((double)i - (double)N * (2*i >= N));
    double freqy = (2.0 * M_PI_F / L) * ((double)j - (double)N * (2*j >= N));
    
    phi[i*N+j] /= a + 0.5f*(freqx*freqx + freqy*freqy);
}

__kernel void sum(__global double2* in_buffer, ulong in_offset, __global double2* out_buffer,  ulong out_offset) {
    int i = get_global_id(0);
    out_buffer[out_offset + i] = in_buffer[in_offset + 2*i] + in_buffer[in_offset + 2*i+1];
}

__kernel void sum_inplace(__global double2* buffer, ulong out_size) {
    int i = get_global_id(0);
    buffer[i] += buffer[i + out_size] ;
}

// Actually there is no SIMD at all...
__kernel void simd(__global double2* buffer, ulong out_size) {
    int i = get_global_id(0);
    double2 s = buffer[i];
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
