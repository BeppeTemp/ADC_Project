#include <stdlib.h>
#include <stdio.h>
#include <math.h>
int **MatrixMultiply(int **A, int **B, int n);
void printMatrix(int **array, int rows, int columns);
int main(void)
{
int i=0;
int j=0;
 
//Stores Matrix A #s
int **A;
A=(int**) malloc(5*sizeof(int*));
for (i=0;i<5;i++)
{
A[i]=(int*) malloc(5*sizeof(int));
A[0][0]=3,A[0][1]=1,A[0][2]=5,A[0][3]=42,A[0][4]=35;
A[1][0]=8,A[1][1]=34,A[1][2]=64,A[1][3]=21,A[1][4]=68;
A[2][0]=61,A[2][1]=41,A[2][2]=55,A[2][3]=32,A[2][4]=87;
A[3][0]=17,A[3][1]=14,A[3][2]=53,A[3][3]=52,A[3][4]=12;
A[4][0]=35,A[4][1]=54,A[4][2]=15,A[4][3]=12,A[4][4]=12;
}
 
 
//Stores Matrix B #s
int **B;
B =(int**) malloc(5*sizeof(int*));
for(j=0; j<5;j++)
{
B[i]=(int*) malloc(5*sizeof(int));
B[0][0]=16,B[0][1]=10,B[0][2]=5,B[0][3]=74,B[0][4]=35;
B[1][0]=71,B[1][1]=8,B[1][2]=68,B[1][3]=38,B[1][4]=98;
B[2][0]=13,B[2][1]=31,B[2][2]=51,B[2][3]=42,B[2][4]=74;
B[3][0]=22,B[3][1]=61,B[3][2]=35,B[3][3]=67,B[3][4]=43;
B[4][0]=54,B[4][1]=47,B[4][2]=12,B[4][3]=11,B[4][4]=33;
}
 
 
 
//prints matrix with for loop
for(i=0;i<5;i++)
{
 printf("%d",**MatrixMultiply(A,B,5));
}
printf("\n");
 
//prints matrix with print function
//printf("%d",printMatrix(C,5,5));
return (0);
}
 
int **MatrixMultiply(int **A, int **B, int n)
{
 
int i;
int j;
int k;
int sum;
int **C;
C=(int**) malloc(5*sizeof(int*));
 
for (i=0;i<5;i++)
{
C[i]=(int*) malloc(5*sizeof(int));
}
 
for (i=0; i<n; i++)
{
 for (j=0; j<n; j++)
 {
  sum = 0;
  for(k=0;k<n;k++)
  {
  sum += A[i][k] * B[k][j];
  }
  C[i][j]=sum;
 
 }
}
 
return(C);
}
 
void printMatrix(int **array, int rows, int columns)
{
  int i, j;
  for(i=0; i<rows; i++)
  {
    printf("[ ");
    for(j=0; j<columns; j++)
    {
        printf("%5d ", array[i][j]);
    }
    printf("]\n");
  }
}