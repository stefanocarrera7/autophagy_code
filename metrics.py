from math import comb

def passatk(n:int, c:int, k:int):
  from math import comb

  k = min(n, k)
  if k <= 0 or n <= 0: return 0.0
  if c <= 0: return 0.0
  if c >= n: return 1.0
  return 1.0 - (comb(n - c, k) / comb(n, k))