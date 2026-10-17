#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(){@autoreleasepool{auto d=MTLCreateSystemDefaultDevice();if(!d)return 2;auto s=d.maxThreadsPerThreadgroup;printf("{\"device\":\"%s\",\"max_threads_per_threadgroup\":[%lu,%lu,%lu],\"scope\":\"Device limit only; specific pipeline resource limit and actual execution still require validation\"}\n",d.name.UTF8String,(unsigned long)s.width,(unsigned long)s.height,(unsigned long)s.depth);}}
