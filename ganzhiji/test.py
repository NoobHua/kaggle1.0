a_dic = {'a': 5, 'b': 6, 'c': 4}
a_dic['a'] =  a_dic.get('a') +1
print(a_dic)
aaa = a_dic.get('d')
if aaa in a_dic.keys():
    print("aaa is not null")
else:
    print('aaa is null')


print(type(aaa) )
if type(aaa) is int:
    print("the type of aaa is int" )
if isinstance(aaa,int):
    print ("the type of aaa is int" )

a = 1
b = 'ccc'
print(a,b)

mydict = {'1.1':5,'1.0':6, '1.5':9, '3.0':20, '20':8}
print(mydict)

sortedClassCount = sorted(mydict.items(), key=lambda d: d[1])
print(int(sortedClassCount[len(sortedClassCount)-1][0]))


