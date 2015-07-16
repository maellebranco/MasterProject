#include <stdio.h>
#include <unistd.h>
#include <time.h>

struct timespec timestamp;

int main()
{
 usleep(50*1000); // wait 50ms before starting
 int i;
 for(i=0; i<200; ++i)
 {
  clock_gettime(CLOCK_REALTIME, &timestamp);
  fprintf(stderr,"%d%09ld\n",(int)timestamp.tv_sec,timestamp.tv_nsec);
  usleep(10*1000); // 10ms
 }
}
