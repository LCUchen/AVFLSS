import random


def judge_prime(p):
    # 素数的判断
    if p <= 1:
        return False
    i = 2
    while i * i <= p:
        if p % i == 0:
            return False
        i += 1
    return True


# 检测大整数是否是素数,如果是素数,就返回True,否则返回False
# rabin算法的意思大家自己百度哈
def rabin_miller(num):
    s = num - 1
    t = 0
    while s % 2 == 0:
        s = s // 2
        t += 1

    for trials in range(5):
        a = random.randrange(2, num - 1)
        v = pow(a, s, num)
        if v != 1:
            i = 0
            while v != (num - 1):
                if i == t - 1:
                    return False
                else:
                    i = i + 1
                    v = (v ** 2) % num
    return True


def is_prime(num):
    # 排除0,1和负数
    if num < 2:
        return False

    # 创建小素数的列表,可以大幅加快速度
    # 如果是小素数,那么直接返回true
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
                    103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
                    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443,
                    449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577,
                    587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
                    709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839,
                    853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983,
                    991, 997]
    if num in small_primes:
        return True

    # 如果大数是这些小素数的倍数,那么就是合数,返回false
    for prime in small_primes:
        if num % prime == 0:
            return False

    # 如果这样没有分辨出来,就一定是大整数,那么就调用rabin算法
    return rabin_miller(num)


def get_prime(seed=0, key_size=1024):
    while True:
        random.seed(seed)
        num = random.randrange(2 ** (key_size - 1), 2 ** key_size)
        if is_prime(num):
            return num
        seed += 1


def multimod(a, k, n):  # 快速幂取模
    ans = 1
    while (k != 0):
        if k % 2:  # 奇数
            ans = (ans % n) * (a % n) % n
        a = (a % n) * (a % n) % n
        k = k // 2  # 整除2
    return ans


def yg(n):  # 求原根
    k = (n - 1) // 2
    list = []
    for i in range(2, n - 1):
        if multimod(i, k, n) != 1:
            list.append(i)
            if random.uniform(0, 1) < .1:
                break
    return list


def get_generator(p):
    # 得到所有的原根
    a = 2
    list = []
    while a < p:
        flag = 1
        while flag != p:
            if (a ** flag) % p == 1:
                break
            flag += 1
        if flag == (p - 1):
            list.append(a)
        a += 1
    return list


# A，B得到各自的计算数
def get_calculation(p, a, X):
    # Y = (a ** X) % p
    Y = fastExpMod(a, X, p)
    return Y


def fastExpMod(b, e, m):
    result = 1
    while e != 0:
        if (e & 1) == 1:
            # ei = 1, then mul
            result = (result * b) % m
        e >>= 1
        # b, b^2, b^4, b^8, ... , b^(2^n)
        b = (b * b) % m
    return result


# A，B得到交换计算数后的密钥
def get_key(X, Y, p):
    # key = (Y ** X) % p
    key = fastExpMod(Y, X, p)
    return key


def dh_exchange(seed):
    p = get_prime(seed)
    # 得到素数的一个原根
    # q = get_generator(p)
    q = yg(p)
    XA = random.randint(0, p - 1)
    # YA = get_calculation(p, int(q[-1]), XA)
    YA = fastExpMod(int(q[-1]), XA, p)
    return XA, YA, p


if __name__ == "__main__":
    # 得到规定的素数
    p = get_prime()

    # 得到素数的一个原根
    print(p, flush=True)
    # q = get_generator(p)
    q = yg(p)
    print(q, flush=True)

    # 得到A的私钥
    XA = random.randint(0, p - 1)
    print('A随机生成的私钥为：%d' % XA)

    # 得到B的私钥
    XB = random.randint(0, p - 1)
    print('B随机生成的私钥为：%d' % XB)
    print('------------------------------------------------------------------------------')

    # 得待A的计算数
    YA = get_calculation(p, int(q[-1]), XA)
    print('A的计算数为：%d' % YA, flush=True)

    # 得到B的计算数
    YB = get_calculation(p, int(q[-1]), XB)
    print('B的计算数为：%d' % YB, flush=True)
    print('------------------------------------------------------------------------------')

    # 交换后A的密钥
    key_A = get_key(XA, YB, p)
    print('A的生成密钥为：%d' % key_A, flush=True)

    # 交换后B的密钥
    key_B = get_key(XB, YA, p)
    print('B的生成密钥为：%d' % key_B)
    print('---------------------------True or False------------------------------------')

    print(key_A == key_B)
