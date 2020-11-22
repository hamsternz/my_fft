#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define FFT_SIZE (64*1024)
#define DFT_SIZE 16

double sines[FFT_SIZE];
double cosines[FFT_SIZE];

double din_r[FFT_SIZE];
double din_i[FFT_SIZE];

double dout_r[FFT_SIZE];
double dout_i[FFT_SIZE];

double dout_r_ref[FFT_SIZE];
double dout_i_ref[FFT_SIZE];

//=================================================================
// Save calling sin() and cos() all the time
//=================================================================
static void init_tables(void) {
   for(int i = 0; i < FFT_SIZE; i++) {
     double phase = 2.0*M_PI*i/FFT_SIZE;
     cosines[i] = cos(phase);
     sines[i]   = sin(phase);
  }
}

static void print_out(double *d_r, double *d_i, int size, char *message) {
   printf("-----------------------------------------\n");
   if(message) {
      printf("%s:\n", message);
   }
   for(int i = 0; i < size; i++) {
      printf("%3i, %10f, %10f\n", i, d_r[i], d_i[i]);
   }
   getchar();
}

static void dft(double *r_i, double *i_i, double *r_o, double *i_o, int size, int stride) {
   for(int bin = 0; bin < size; bin++) {
      int i = 0;
      double total_i = 0.0;
      double total_q = 0.0;
      for(int s = 0; s < size; s++) {
         total_i +=  r_i[s*stride] * cosines[i] - i_i[s*stride] *   sines[i];
         total_q +=  r_i[s*stride] *   sines[i] + i_i[s*stride] * cosines[i];
         i += bin*(FFT_SIZE/size);
         if(i >= FFT_SIZE) i -= FFT_SIZE;
      }
      r_o[bin] = total_i/FFT_SIZE;
      i_o[bin] = total_q/FFT_SIZE;
   }
}

static int reverse_bits(int i, int max) {
   int o = 0;
   assert(i < max || i >= 0);
   i |= max;
   while(i != 1) {
      o <<= 1;
      if(i&1) 
        o |= 1; 
      i >>= 1;
   }
   return o;
}

static void fft_v2(double *r_i, double *i_i, double *r_o, double *i_o, int size) {
    int i, stride, step;

    stride = size/DFT_SIZE;
    for(i = 0; i < stride ; i++) {
        int out_offset = reverse_bits(i,stride) * DFT_SIZE;
        dft(r_i+i, i_i+i,  r_o+out_offset, i_o+out_offset, DFT_SIZE, stride);
    }

    stride = DFT_SIZE*2;
    step = FFT_SIZE/stride;
    while(stride <= FFT_SIZE) {
       for(i = 0; i < FFT_SIZE; i+= stride) {
          double *real = r_o+i;
          double *imag = i_o+i;
          size = stride/2;
          for(int j = 0; j < size; j++) {
             double c = cosines[j*step], s = sines[j*step];
             double rotated_r =  real[size] * c - imag[size] * s;
             double rotated_i =  real[size] * s + imag[size] * c;
             real[size] = real[0] - rotated_r;
             imag[size] = imag[0] - rotated_i;
             real[0]    = real[0] + rotated_r;
             imag[0]    = imag[0] + rotated_i;
             real++;
             imag++;
          }
       }
       stride *= 2;
       step   /= 2;
    }
}

static void fft_v1(double *r_i, double *i_i, double *r_o, double *i_o, int size, int stride) {
    int half_size = size/2;

    if(half_size >  DFT_SIZE && (half_size & 1) == 0) {
       fft_v1(r_i,        i_i,        r_o,            i_o,           half_size, stride*2);
       fft_v1(r_i+stride, i_i+stride, r_o+half_size,  i_o+half_size, half_size, stride*2);
    } else {
       dft(r_i,        i_i,        r_o,            i_o,           half_size, stride*2);
       dft(r_i+stride, i_i+stride, r_o+half_size,  i_o+half_size, half_size, stride*2);
    }

    int step = FFT_SIZE/size;
    for(int i = 0; i < half_size; i++) {
       double c = cosines[i*step];
       double s =   sines[i*step];

       double even_r = r_o[i];
       double even_i = i_o[i];
       double odd_r  = r_o[i+half_size];
       double odd_i  = i_o[i+half_size];

       double rotated_r =  odd_r * c - odd_i * s;
       double rotated_i =  odd_r * s + odd_i * c;
       r_o[i]           = even_r + rotated_r;
       i_o[i]           = even_i + rotated_i;
       r_o[i+half_size] = even_r - rotated_r;
       i_o[i+half_size] = even_i - rotated_i;
    }
}

void ts_sub(struct timespec *r, struct timespec *a, struct timespec *b) {
   r->tv_sec = a->tv_sec - b->tv_sec;
   if(a->tv_nsec > b->tv_nsec) {
      r->tv_nsec = a->tv_nsec - b->tv_nsec;
   } else {
      r->tv_sec--;
      r->tv_nsec = a->tv_nsec - b->tv_nsec+1000000000;
   }
}

void check_error(double *r_ref, double *i_ref, double *r, double *j, int size) {
   // Check for error
   double error = 0.0;
   for(int j = 0; j < FFT_SIZE; j++) {
      error += fabs(dout_r_ref[j] - dout_r[j]); 
      error += fabs(dout_i_ref[j] - dout_i[j]); 
   };
   printf("  Total error is %10e\n",error);
}

int main(int argc, char *argv[]) {
   struct timespec tv_start_dft,    tv_end_dft,    tv_dft;
   struct timespec tv_start_fft_v1, tv_end_fft_v1, tv_fft_v1;
   struct timespec tv_start_fft_v2, tv_end_fft_v2, tv_fft_v2;

   // Setup
   for(int i = 0; i < FFT_SIZE; i++) {
#if 0
      if(i %64 < 32)  
         din_r[i] =   1.0;
      else
         din_r[i] =  -1.0;
      din_i[i] = 0;
#else
      din_r[i] =rand() %101 /100.0;
      din_i[i] =rand() %101 /100.0;
#endif
   }
   init_tables();

   printf("Transform of %5i random complex numbers\n", FFT_SIZE);
   printf("=========================================\n");

   clock_gettime(CLOCK_MONOTONIC, &tv_start_dft);  
   dft(din_r,din_i, dout_r_ref, dout_i_ref, FFT_SIZE,1);
   clock_gettime(CLOCK_MONOTONIC, &tv_end_dft);  

   ts_sub(&tv_dft,    &tv_end_dft,    &tv_start_dft);
   printf("DFT %u.%09u secs\n",(unsigned)tv_dft.tv_sec, (unsigned)tv_dft.tv_nsec);

   clock_gettime(CLOCK_MONOTONIC, &tv_start_fft_v1);  
   fft_v1(din_r,din_i, dout_r,     dout_i, FFT_SIZE, 1);
   clock_gettime(CLOCK_MONOTONIC, &tv_end_fft_v1);  

   ts_sub(&tv_fft_v1, &tv_end_fft_v1, &tv_start_fft_v1);
   printf("FFT recursive %u.%09u secs\n",(unsigned)tv_fft_v1.tv_sec, (unsigned)tv_fft_v1.tv_nsec);
   check_error(dout_r_ref, dout_i_ref, dout_r,dout_i, FFT_SIZE);

   clock_gettime(CLOCK_MONOTONIC, &tv_start_fft_v2);  
   fft_v2(din_r,din_i, dout_r,     dout_i, FFT_SIZE);
   clock_gettime(CLOCK_MONOTONIC, &tv_end_fft_v2);  

   ts_sub(&tv_fft_v2, &tv_end_fft_v2, &tv_start_fft_v2);
   printf("FFT looped    %u.%09u secs\n",(unsigned)tv_fft_v2.tv_sec, (unsigned)tv_fft_v2.tv_nsec);
   check_error(dout_r_ref, dout_i_ref, dout_r,dout_i, FFT_SIZE);

   (void)print_out;  
}

