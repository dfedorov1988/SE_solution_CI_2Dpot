/* testlib.c */
/* Compilation requires GNU scientific math library. Compilation should look like this:*/
/* gcc -L/usr/local/lib -I/usr/local/include -lgsl -lgslcblas -shared -o V11.so V11.c */
#include </Users/Dmitry/gsl-2.4/gsl/gsl_sf_bessel.h>
#include <stdio.h>
double f(int m, double *x, void *user_data) {
    double *c = (double *)user_data;
	return gsl_sf_bessel_Jn(c[0],gsl_sf_bessel_zero_Jnu(c[0],c[1])*x[0]/c[4])\
	* gsl_sf_bessel_Jn(c[2],gsl_sf_bessel_zero_Jnu(c[2],c[3])*x[0]/c[4])\
	/ (c[4]*gsl_sf_bessel_Jn(c[2]+1,gsl_sf_bessel_zero_Jnu(c[2],c[3])))\
	/ (c[4]*gsl_sf_bessel_Jn(c[0]+1,gsl_sf_bessel_zero_Jnu(c[0],c[1])))*(x[0]*x[0]+c[5]*c[5]/4)*x[0];
}
