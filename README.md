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
