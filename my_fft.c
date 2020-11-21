#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>

#define FFT_SIZE (1024)
#define DO_FFT  8
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

static void fft(double *r_i, double *i_i, double *r_o, double *i_o, int size, int stride) {
    int half_size = size/2;

    if(half_size >=  DO_FFT && (half_size & 1) == 0) {
       fft(r_i,        i_i,        r_o,            i_o,           half_size, stride*2);
       fft(r_i+stride, i_i+stride, r_o+half_size,  i_o+half_size, half_size, stride*2);
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

int main(int argc, char *argv[]) {
   struct timespec tv_start, tv_middle, tv_end, tv_fft, tv_dft;

   // Setup
   for(int i = 0; i < FFT_SIZE; i++) {
#if 1
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

   // The time sensitive bits
   clock_gettime(CLOCK_MONOTONIC, &tv_start);  
   fft(din_r,din_i, dout_r,     dout_i, FFT_SIZE,1);
   clock_gettime(CLOCK_MONOTONIC, &tv_middle);  
   dft(din_r,din_i, dout_r_ref, dout_i_ref, FFT_SIZE,1);
   clock_gettime(CLOCK_MONOTONIC, &tv_end);  

   tv_fft.tv_sec = tv_middle.tv_sec - tv_start.tv_sec;
   if(tv_middle.tv_nsec > tv_start.tv_nsec) {
      tv_fft.tv_nsec = tv_middle.tv_nsec - tv_start.tv_nsec;
   } else {
      tv_fft.tv_sec--;
      tv_fft.tv_nsec = tv_middle.tv_nsec - tv_start.tv_nsec+1000000000;
   }

   tv_dft.tv_sec = tv_end.tv_sec - tv_middle.tv_sec;
   if(tv_end.tv_nsec > tv_middle.tv_nsec) {
      tv_dft.tv_nsec = tv_end.tv_nsec - tv_middle.tv_nsec;
   } else {
      tv_dft.tv_sec--;
      tv_dft.tv_nsec = tv_end.tv_nsec - tv_middle.tv_nsec+1000000000;
   }

   // Print results
//   print_out(dout_r,     dout_i,     FFT_SIZE, "FFT");
//   print_out(dout_r_ref, dout_i_ref, FFT_SIZE, "DFT reference");
   printf("Transform of %5i random complex numbers\n", FFT_SIZE);
   printf("=========================================\n");
   printf("DFT %u.%09u\n",(unsigned)tv_dft.tv_sec, (unsigned)tv_dft.tv_nsec);
   printf("FFT %u.%09u\n",(unsigned)tv_fft.tv_sec, (unsigned)tv_fft.tv_nsec);

   // Check for error
   double error = 0.0;
   for(int i = 0; i < FFT_SIZE; i++) {
      error += fabs(dout_r_ref[i] - dout_r[i]); 
      error += fabs(dout_i_ref[i] - dout_i[i]); 
   };
   printf("Total error is %10f\n",error);
   (void)print_out;  
}

