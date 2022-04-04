# arc

## Training policy and value networks
- Pretrain policy network using loss fn that minimizes loss over desired action sequences:
```
  L_pt(policy) = E[sum{t <= T} log policy(action_t | state, spec)]
               = E[sum{t <= T} log policy(d_t | f', B', B)]
```
- Train value function by sampling rollouts from the policy network and computing heuristic value functions offline:
```
  L_train(policy, value) = R * sum{t <= T} log value(state_t, spec) 
                           + (1 - R) * sum{t <= T} log(1 - value(state_t, spec))
                           + R * sum{t <= T} log policy(a_t | state_t, spec)
                         = R * sum{t <= T} log value(f', B', B) 
                           + (1 - R) * sum{t <= T} log(1 - value(f', B', B))
                           + R * sum{t <= T} log policy(d_t | f', B', B)

```

## Inference of z's vs generation of heterogeneous z's
1. add heterogeneous z's:
   ```
    let z = rand_vec(n, 2)
    let z1, z2 = split(z)
    yield (B_z1, B_z1', f', d), (B_z1, B_z2', f', d), (B_z2, B_z1', f', d), (B_z2, B_z2', f', d) 
   ```
2. incorporate inference of z's into NN

## Problems
- value function expects fixed-size inputs, but this doesn't play nice with program gen
  - add padding before passing sth into the value fn? 