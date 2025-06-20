import secretflow as sf

# Replace with the `node-ip-address` and `port` of head node.
sf.init(parties=['alice', 'bob'], address='ip:port')
alice = sf.PYU('alice')
bob = sf.PYU('bob')
print(alice(lambda x : x)(2))
print(bob(lambda x : x)(2))