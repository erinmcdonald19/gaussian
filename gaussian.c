/* Gaussian Elimination
 * Erin McDonald and Michael Franklin
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int MAX_THREADS = 64;

void Upper_triangular(double A[], double b[], int n, int thread_count);
void Row_solve(double A[], double b[], double x[], int n, int thread_count, double tmp);
void usage(char* prog_name);

int main(int argc, char * argv[]) {

    // Ensure correct command line args
    if (argc != 3){
        usage(argv[0]);
    }

    char *f_name = argv[1];
    int threadCount = (int) strtol(argv[2], NULL, 10);

    if (threadCount <= 0 || threadCount > MAX_THREADS){
        usage(argv[0]);
    }

    // Read in file information

    FILE *in_file; 
    
    if((in_file = fopen(f_name, "r")) == NULL){
    	perror("Error");
    }

    int rows;
    fscanf(in_file, "%d", &rows);
    int cols;
    fscanf(in_file, "%d", &cols);

    double * A = malloc(rows*cols*sizeof(double));
    double * b = malloc(rows*sizeof(double));
    double * x = malloc(rows*sizeof(double));

    int i, j;
    for(i = 0; i < rows; i++){
    	for(j = 0; j < cols; j++){
    		fscanf(in_file, "%lf", &A[i*cols + j]);
    	}
    }
    for(i=0; i < rows; i++){
	fscanf(in_file, "%lf", &b[i]);
    }

    double tmp = 0;

    // Upper_triangular and Row_solve are the two steps required for gaussian elimination

    Upper_triangular(A, b, rows, threadCount);

    Row_solve(A, b, x, rows, threadCount, tmp);

    //print solution

    printf("The solution vector is: \n");
    for(i=0; i< rows; i++){
	printf("%lf\n", x[i]);
    }
    
    //free memory

    free(A);
    free(b);
    free(x);


    return 0;
}


/*--------------------------------------------------------------------
 * Function:  Upper_triangular
 * Purpose:   Convert a matrix into upper triangular format to prepare for back substitution
 * In args:   A, b, n, thread_count
 *
 */
void Upper_triangular(double *A, double *b, int n, int thread_count) {
	int i,j,k;  
//create threads before first loop to avoid overhead of repeatedly creating/destroying
#pragma omp parallel num_threads(thread_count) default(none) shared(A,b,n) private(i,j,k)
	for(i = 0; i < n-1; i++){
//parallelize j loop since there are no dependencies 
#pragma omp for 
		for(j = i+1; j < n; j++) {
            // get the ratio
			double r = A[j*n+i] / A[i*n+i];
			for(k = i; k < n; k++) {
				A[j*n+k] -= (r * A[i*n+k]);	
			}
			b[j] -= (r * b[i]);
		}
#pragma omp barrier
	}
	return;
} /* Upper_triangular */


/*--------------------------------------------------------------------
 * Function:  Row_solve
 * Purpose:   Solve a triangular system using the row-oriented algorithm
 * In args:   A, b, n, thread_count, tmp
 * Out arg:   x
 *
 * Notes: Written as a class
 */
void Row_solve(double *A, double *b, double *x, int n, int thread_count, double tmp) {
   	int i, j;

#pragma  omp parallel num_threads(thread_count) default(none) private(i, j) shared(A, b, x, n, tmp)  
	for (i = n-1; i >= 0; i--) {
#pragma omp single
      		tmp = b[i];

#pragma omp for reduction(+: tmp)
     		for (j = i+1; j < n; j++)
         		tmp += -A[i*n+j]*x[j];

#pragma omp single
     		x[i] = tmp/A[i*n+i];
   }
   return;
}  /* Row_solve */


void usage(char* prog_name) {
    fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
    fprintf(stderr, "0 < number of threads <= %d\n", MAX_THREADS);
    exit(0);
}  /* Usage */
