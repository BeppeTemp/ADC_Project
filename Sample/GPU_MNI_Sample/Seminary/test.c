#include <stdio.h>
#include <stdlib.h>

int main() {
    for (int i = 0; i < 10; i++) {
        /* code */

        float num =  rand() % 4 - 2;
        printf("%f\n", num);
    }
    return 0;
}