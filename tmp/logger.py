

import logging
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') 
logging.basicConfig(
    level=logging.DEBUG,
    filename='./tmp/'+'log.txt',
    filemode='w',
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt = '%Y-%m-%d  %H:%M:%S %a'
)
logger=logging.getLogger(__name__)
logger.setLevel('INFO')

class A():
    def __init__(self,b):
        self.b=b

    def log(self):
        logger.info("I'm logger"+str(self.b))

    def log2(self):
        logger.info(msg='Iteration'.ljust(25) + '%f' % self.b)


if __name__ == "__main__":
    a=A(12)
    a.log()
    a.log2()

    
