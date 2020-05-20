#include <stdio.h>

struct aa{
    int m;
    int n;
    char a;
};

int main()
{
    struct aa test;
   printf("  %d  %d  \n",sizeof(struct aa),(int)&test.n-(int)&test );
    return 1;
}
