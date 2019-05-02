import copy
import commands

def brute():
    pass

def rotcat(matrix):
    ret=[]
    for i in xrange(6):
        v=0
        for j in xrange(3):
            for k in xrange(3):
                if j==k==1:continue
                v=v*6+matrix[i][j][k]
        ret.append(v)
    return tuple(ret)

def pick(rotmats, strage):
    maxpos=0
    maxval=(0,0)
    for rot in rotmats:
        for sind,ss in enumerate(strage):
            match = 1
            for ind, s in enumerate(ss[1:]):
                if rotmats[rot][s[0]][s[1]][s[2]] != s[0]: 
                    match = 0
                    break
            if match==0: break
        if sind > maxpos or (sind==maxpos and maxval>rot):
            #print 'match', sind, rot
            maxval=rot
            maxpos = sind
    ret=''
    rotcmd=maxval[0]
    for i in xrange(maxval[1]):
        ret='!#$^&('[rotcmd%6] if (rotcmd%12)>=6 else '134679'[rotcmd%6] +ret
        rotcmd/=12
    return ret
    
def run(plane, adjustFn):
    cind={}
    for i in xrange(6):
        cind[plane[i][1][1]]=i
    ts='UFRBLD'
    order=[[(0,2,1),(1,0,1)],[(0,1,2),(2,0,1)],[(0,0,1),(3,0,1)],[(0,1,0),(4,0,1)],
           [(5,0,1),(1,2,1)],[(5,1,2),(2,2,1)],[(5,2,1),(3,2,1)],[(5,1,0),(4,2,1)],
           [(1,1,2),(2,1,0)],[(1,1,0),(4,1,2)],[(3,1,0),(2,1,2)],[(3,1,2),(4,1,0)],
           [(0,2,2),(1,0,2),(2,0,0)],[(0,0,2),(2,0,2),(3,0,0)],[(0,0,0),(3,0,2),(4,0,0)],[(0,2,0),(4,0,2),(1,0,0)],
           [(5,0,2),(2,2,0),(1,2,2)],[(5,0,0),(1,2,0),(4,2,2)],[(5,2,0),(4,2,0),(3,2,2)],[(5,2,2),(3,2,0),(2,2,2)],
           ]
    param = ''
    for block in order:
        for p in block:
            param+=ts[cind[plane[p[0]][p[1]][p[2]]]]
        param+=' '
    #print 'param', param
    outstat, retcmd = commands.getstatusoutput('./a.out %s'%param)
    #print outstat, retcmd
    cmdmap={'U':'4','F':'!','B':'3','D':'^','L':'9','R':'&'}
    cmdrmp={'U':'$','F':'1','B':'#','D':'6','L':'(','R':'7'}
    ret=''
    print 'done', retcmd
    for c in retcmd.split():
        if len(c)==1:
            ret+=cmdmap[c[0]]
        elif c[1]=='3':
            ret+=cmdrmp[c[0]]
        else:
            ret+=cmdmap[c[0]]*int(c[1])
    #print 'done', retcmd, ret
    return ret 
def run0(plane, adjustFn):
    #print 'think...', plane
    cind={}
    for i in xrange(6):
        cind[plane[i][1][1]]=i
    newp = []
    for i in xrange(6):
        newp.append([])
        for j in xrange(3):
            newp[i].append([cind[k] for k in plane[i][j]])
    #print newp
    strage=[['brute',(0,2,1),(1,0,1)],['brute',(2,1,0),(1,1,2)],['brute',(5,0,1),(1,2,1)],['brute',(4,1,2),(1,1,0)],
            ['brute',(0,2,0),(1,0,0)],['brute',(2,0,0),(1,0,2)],['brute',(4,2,2),(1,2,0)],['brute',(2,2,0),(1,2,2)],
            ['',(4,0,1),(0,1,0)],['',(2,0,1),(0,1,2)],['',(2,2,1),(5,1,2)],['',(5,1,0),(4,2,1)],
            ['',(4,0,1),(0,1,0)],['',(2,0,1),(0,1,2)],['',(2,2,1),(5,1,2)],['',(5,1,0),(4,2,1)],
            ['',(4,0,1),(0,1,0)],['',(2,0,1),(0,1,2)],['',(2,2,1),(5,1,2)],['',(5,1,0),(4,2,1)],
            ]
    rotfmt={}
    rotres=''
    chgcache={}
    skipset=set()
    rotchar=set()
    chgcache[(0, 0)]=newp

    for rotlen in xrange(1,6):
        for r in xrange(12**rotlen):
            if (r/12, rotlen-1) in skipset:
                #print 'skip', r, rotlen
                skipset.add((r, rotlen))
                continue
            #print r, rotlen
            cmd=[0,2,3,5,6,8][r%6]
            tempp=copy.deepcopy(chgcache[(r/12, rotlen-1)])
            adjustFn(tempp, cmd, r>=6)
            v=rotcat(tempp)
            if v in rotchar:
                #print 'skip1', r, rotlen
                skipset.add((r,rotlen))
                continue
            rotchar.add(v)
            chgcache[(r, rotlen)]=tempp

    ret = pick(chgcache, strage)
    print 'calc over'
    return ret

if __name__ == "__main__":
    import viewer
    v=viewer.Viewer()
    plane=[[[0,0,0],[0,0,0],[4,4,4]],[[1,1,1],[1,1,1],[1,1,1]],[[0,2,2],[0,2,2],[0,2,2]],
            [[3,3,3],[3,3,3],[3,3,3]],[[4,4,5],[4,4,5],[4,4,5]], [[2,2,2],[5,5,5],[5,5,5]]]
    print run(plane, v.adjustPlan)
