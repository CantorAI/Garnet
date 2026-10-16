"""Same scalar/vector shaders and numerical gates; prospective fixed timing design only."""
from make_vector_vision_context import generate as original_generate

def generate(native,harness,selected_header):
    shader,cpp=original_generate(native,harness,selected_header)
    for old,new in [('count()<.25','count()<2.0'),('round<2','round<10'),('trial<20','trial<5'),
                    (r'\"samples\":20',r'\"samples\":5'),
                    ('(gpu[9]+gpu[10])/2','gpu[2]'),('(wall[9]+wall[10])/2','wall[2]')]:
        assert cpp.count(old)==1,(old,cpp.count(old));cpp=cpp.replace(old,new)
    return shader,cpp
