import math
from shapely import Polygon, LineString
from matplotlib import pyplot as plt

class Mock:
        
    def __init__(self, x, y, Y):
          self.x = x
          self.y = y
          self.yaw = Y

b = 2
f = 12
s = 3.5

n1 = Mock(1,1,math.radians(30))
n2 = None # Mock(5,5,math.atan2(4,4))

def get_angle(c, h):
      
      ip = [c[0] + (10 * math.cos(h) / 2), c[1] + (10 * math.sin(h) / 2)]

      return LineString((c, ip))


def create_padding_box(b, f, s, n1, n2=None):

        st, ed = n1, n2 if n2 is not None else n1
        t = ed.yaw

        p1 = [st.x - b*math.cos(t) + s*math.sin(t), st.y - b*math.sin(t) - s*math.cos(t)]
        p2 = [st.x - b*math.cos(t) - s*math.sin(t), st.y - b*math.sin(t) + s*math.cos(t)]
        p3 = [ed.x + f*math.cos(t) - s*math.sin(t), ed.y + f*math.sin(t) + s*math.cos(t)]
        p4 = [ed.x + f*math.cos(t) + s*math.sin(t), ed.y + f*math.sin(t) - s*math.cos(t)]

        pol = Polygon((p1, p2, p3, p4))

        return pol, LineString(([st.x, st.y], [ed.x, ed.y]))

a, b = create_padding_box(b, f, s, n1, n2)

plt.plot(*a.exterior.xy)

plt.plot(*b.xy)

plt.show()