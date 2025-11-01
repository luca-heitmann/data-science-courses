
import random
import numpy as np
import string


class weirdlysortedtuple():
    def __init__(self):
        #inits it as an empty tuple
        self.data = ()
    
    def __init__(self,el1,el2):
        self.data = (el1, el2)

    #https://docs.python.org/3/reference/datamodel.html#object.__str__
    def __repr__(self):
        return f"({self.data[0]}, {self.data[1]})"

    
    def __lt__(self,other):
        return (self.data[0] < other.data[0]) or (self.data[0] == other.data[0] and self.data[1] > other.data[1])

 

def testrun():

  seed  =3 
  random.seed(seed)
  rng = np.random.default_rng(seed)
  
  strlen=5  
  numel = 15

  tosort1=[]  
  tosort2=[]
  for _ in range(numel):

    a = rng.integers(low=9, high=13)

    #try out both
    charlist = random.SystemRandom().choices(string.ascii_lowercase, k=strlen)
    #charlist = random.choices(string.ascii_lowercase, k=strlen)
    tmp = ''.join(charlist)
    
    tosort1.append((a,tmp))
    tosort2.append(  weirdlysortedtuple(a,tmp))

  #these should NOT match
  print(tosort1)

  b= sorted(tosort1)
  print('standard sort order:\n', b)

  a= sorted(tosort2)
  print('counterintuitive sort order:\n', a)



if __name__=='__main__':
  testrun()
