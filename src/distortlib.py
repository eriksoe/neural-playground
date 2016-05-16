# Library to distort sounds.

import random

def scale(samples, amount):
    coeff = 1 + amount * random.uniform(-0.5, 0.2)
    return [x * coeff for x in samples]

def nonlinear1(samples, amount):
    coeff = 20*amount
    return [x + coeff * x**3 for x in samples]

def nonlinear2(samples, amount):
    coeff = 5*amount
    return [x + coeff * x**2 for x in samples]

def add_white_noise(samples, amount):
    coeff = 0.05 * amount
    return [x + coeff * random.uniform(0,1) for x in samples]

def distort_time(samples, amount):
    t=0
    min_dt = 1 - 0.15 * amount
    max_dt = 1 + 0.15 * amount
    dt = 1
    fixedness = 0
    res = []
    while t < len(samples):
        res.append(samples[int(t)])
        if random.uniform(0,1) < 0.01:
            rnd_dt = random.uniform(min_dt, max_dt)
            dt = fixedness*dt + (1-fixedness)*rnd_dt
        t += dt
    return res

def distort_time2(samples, amount):
    size = len(samples)
    res = [0 for s in range(size)]
    #res[0] = samples[0]
    #res[size-1] = samples[size-1]
    distort_time_rec(samples, res, 0.0,float(size), 0.0,float(size), 0.1 * amount)
    return res

def distort_time_rec(src, dest, src_a, src_b, dest_a, dest_b, amount):
    #print "Range: [%s ; %s] of %s <- [%s ; %s] of %s" % (dest_a, dest_b, len(dest), src_a, src_b, len(src))
    if src_a > src_b:
        raise Exception("Bad interval")
    if dest_b - dest_a < 1:
        dest[int(dest_a)] = src[int(src_a)]
    else:
        dest_mid = (dest_a + dest_b) / 2
        src_mid = (src_a + src_b) / 2
        src_rnd = random.uniform(src_a, src_b)
        src_mid = src_mid * (1-amount) + src_rnd * amount # Offset src midpoint
        if src_mid > src_b:
            print "Gah! (%s,%s) -> %s, %s" % (src_a, src_b, src_rnd, src_mid)
        distort_time_rec(src, dest, src_a, src_mid, dest_a, dest_mid, amount)
        distort_time_rec(src, dest, src_mid, src_b, dest_mid, dest_b, amount)

distortion_collection = [
    scale,
    nonlinear1, nonlinear2,
    add_white_noise,
    distort_time, distort_time2
]
