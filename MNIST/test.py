def a(p1):
    print(p1)
    i = 999
    print(i)
    def b(p2):
        print(p2*i)
    b(p1)
    i += 1
    b(p1)

a(2)
c = 0
c
c