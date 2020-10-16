import os
import commands
proto_path = os.path.join(os.getcwd(),"prototxts")
proto_list = os.listdir(proto_path)
format = "{:20s}{:20s}{:20s}{:20s}"
print format.format('name','cpu/mlu','cpu/mfus','mlu/mfus')
for proto in proto_list:
    name = proto.split('.')[0]
    command = './test_op.sh prototxts/{} {}'.format(proto,name+'.caffemodel')
    _,result = commands.getstatusoutput(command)
    result = result.strip().split('\n')
    errs = [ line for line in result if line.startswith('errRate') ]
    if len(errs) == 3:
        scrap = lambda x:x.split('=')[1].strip()
        print format.format(name,scrap(errs[0]),scrap(errs[1]),scrap(errs[2]))
    else:
        print 'work on {} failed!'.format(name)
    os.system("rm {}.caffemodel".format(name))
    os.system("rm {}_log*".format(name))
