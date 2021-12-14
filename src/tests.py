from grammar import *
import viz
import random

def rect(x, y, w, h, color):
    return Rect(Z(x), Z(y), 
                Plus(Z(x), Num(w)), Plus(Z(y), Num(h)), 
                color=Z(color))

if __name__ == '__main__':
    # s = e.serialize(); # print(s)
    # d = deserialize(s); print(d)

    n = 10
    env = {'z': seed_zs(), 
           'sprites': seed_sprites()}
    sprites = [rand_sprite(i, env) for i in range(n)] 
    sprites = simplify(sprites, env)
    e = Seq(*sprites)
    print(e)
    b = e.eval(env); print(b)
    viz.viz_mult([b] + [s.eval(env) for s in sprites],
                 text='\n'.join([str(s) for s in sprites]))

