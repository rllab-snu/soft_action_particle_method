[200~#include <stdio.h>
#include <time.h>
#include <math.h>

#include "timer.h"

void sleep_sec(double t) {
	  struct timespec ts;
	    ts.tv_sec = (long)t;
	      ts.tv_nsec = (long)(fmod(t, 1) * 1e9);
	        nanosleep(&ts, NULL);
}

static int fail_count = 0;

void compare_time(const char *name, double result, double expected) {
	  if (fabs(result - expected) < 1e-2) { // 10ms error
		      printf("Test %s SUCCESS (result=%f, expected=%f)\n", name, result, expected);
		        } else {
				    printf("Test %s FAIL    (result=%f, expected=%f)\n", name, result, expected);
				        ++fail_count;
					  }
}

int main() {
	  const double INTERVAL = 0.1;
	    const int IDX0 = 2, IDX1 = 42, IDX2 = 777;

	      // Test 1: single timer
	      //   timer_init(1);
	      //     sleep_sec(INTERVAL);
	      //       compare_time("1-1", timer_read(0), 0);
	      //         timer_start(0);
	      //           sleep_sec(INTERVAL);
	      //             compare_time("1-2", timer_read(0), INTERVAL);
	      //               sleep_sec(INTERVAL);
	      //                 compare_time("1-3", timer_read(0), INTERVAL * 2);
	      //                   sleep_sec(INTERVAL);
	      //                     compare_time("1-4", timer_read(0), INTERVAL * 3);
	      //                       timer_stop(0);
	      //                         sleep_sec(INTERVAL);
	      //                           compare_time("1-5", timer_read(0), INTERVAL * 3);
	      //                             timer_finalize();
	      //
	      //                               // Test 2: multiple timer
	      //                                 timer_init(999);
	      //                                   sleep_sec(INTERVAL);
	      //                                     compare_time("2-1", timer_read(IDX0), 0);
	      //                                       compare_time("2-2", timer_read(IDX1), 0);
	      //                                         compare_time("2-3", timer_read(IDX2), 0);
	      //                                           timer_start(IDX0);
	      //                                             timer_start(IDX1);
	      //                                               timer_start(IDX2);
	      //                                                 sleep_sec(INTERVAL);
	      //                                                   compare_time("2-4", timer_read(IDX0), INTERVAL);
	      //                                                     compare_time("2-5", timer_read(IDX1), INTERVAL);
	      //                                                       compare_time("2-6", timer_read(IDX2), INTERVAL);
	      //                                                         timer_stop(IDX1);
	      //                                                           sleep_sec(INTERVAL);
	      //                                                             compare_time("2-7", timer_read(IDX0), INTERVAL * 2);
	      //                                                               compare_time("2-8", timer_read(IDX1), INTERVAL);
	      //                                                                 compare_time("2-9", timer_read(IDX2), INTERVAL * 2);
	      //                                                                   timer_finalize();
	      //
	      //                                                                     // Test 3: reset check
	      //                                                                       timer_init(999);
	      //                                                                         timer_start(IDX1);
	      //                                                                           sleep_sec(INTERVAL);
	      //                                                                             timer_reset(IDX1);
	      //                                                                               compare_time("3-1", timer_read(IDX1), 0);
	      //                                                                                 timer_start(IDX1);
	      //                                                                                   sleep_sec(INTERVAL);
	      //                                                                                     timer_stop(IDX1);
	      //                                                                                       timer_reset(IDX1);
	      //                                                                                         compare_time("3-2", timer_read(IDX1), 0);
	      //                                                                                           timer_finalize();
	      //
	      //                                                                                             // Test 4: exceptional cases
	      //                                                                                               timer_init(999);
	      //                                                                                                 timer_start(IDX1);
	      //                                                                                                   sleep_sec(INTERVAL);
	      //                                                                                                     timer_start(IDX1);
	      //                                                                                                       sleep_sec(INTERVAL);
	      //                                                                                                         compare_time("4-1", timer_read(IDX1), INTERVAL * 2);
	      //                                                                                                           timer_stop(IDX1);
	      //                                                                                                             timer_stop(IDX1);
	      //                                                                                                               compare_time("4-2", timer_read(IDX1), INTERVAL * 2);
	      //                                                                                                                 timer_start(IDX1);
	      //                                                                                                                   sleep_sec(INTERVAL);
	      //                                                                                                                     timer_start(IDX1);
	      //                                                                                                                       sleep_sec(INTERVAL);
	      //                                                                                                                         compare_time("4-3", timer_read(IDX1), INTERVAL * 4);
	      //                                                                                                                           timer_stop(IDX1);
	      //                                                                                                                             timer_stop(IDX1);
	      //                                                                                                                               compare_time("4-4", timer_read(IDX1), INTERVAL * 4);
	      //                                                                                                                                 timer_finalize();
	      //
	      //                                                                                                                                   // Test 5: skew start
	      //                                                                                                                                     timer_init(999);
	      //                                                                                                                                       timer_start(IDX1);
	      //                                                                                                                                         sleep_sec(INTERVAL);
	      //                                                                                                                                           timer_start(IDX2);
	      //                                                                                                                                             sleep_sec(INTERVAL);
	      //                                                                                                                                               timer_stop(IDX2);
	      //                                                                                                                                                 sleep_sec(INTERVAL);
	      //                                                                                                                                                   timer_stop(IDX1);
	      //                                                                                                                                                     compare_time("5-1", timer_read(IDX1), INTERVAL * 3);
	      //                                                                                                                                                       compare_time("5-2", timer_read(IDX2), INTERVAL * 1);
	      //                                                                                                                                                         timer_finalize();
	      //
	      //                                                                                                                                                           // Test 6: stop and restart, reset and restart
	      //                                                                                                                                                             timer_init(999);
	      //                                                                                                                                                               timer_start(IDX1);
	      //                                                                                                                                                                 sleep_sec(INTERVAL);
	      //                                                                                                                                                                   timer_stop(IDX1);
	      //                                                                                                                                                                     sleep_sec(INTERVAL);
	      //                                                                                                                                                                       timer_start(IDX1);
	      //                                                                                                                                                                         sleep_sec(INTERVAL);
	      //                                                                                                                                                                           compare_time("6-1", timer_read(IDX1), INTERVAL * 2);
	      //                                                                                                                                                                             timer_reset(IDX1);
	      //                                                                                                                                                                               sleep_sec(INTERVAL);
	      //                                                                                                                                                                                 timer_start(IDX1);
	      //                                                                                                                                                                                   sleep_sec(INTERVAL);
	      //                                                                                                                                                                                     compare_time("6-2", timer_read(IDX1), INTERVAL);
	      //                                                                                                                                                                                       timer_finalize();
	      //
	      //                                                                                                                                                                                         if (fail_count == 0) {
	      //                                                                                                                                                                                             printf("ALL SUCCESS!\n");
	      //                                                                                                                                                                                               } else {
	      //                                                                                                                                                                                                   printf("FAILED IN %d CASES T_T\n", fail_count);
	      //                                                                                                                                                                                                       return 1;
	      //                                                                                                                                                                                                         }
	      //
	      //                                                                                                                                                                                                           return 0;
	      //                                                                                                                                                                                                           }
	      //
