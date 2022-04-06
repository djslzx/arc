# TODO
- [ ] test policy net
- [ ] incorporate inference of random parameters (z) 
- [ ] sample rollouts from policy
  ```
    program := empty program
      while n_tokens(program) < self.max_p_len:
        delta := [LINE_START]
        while delta not complete:
          next_delta <- self.forward(bitmaps, program.eval(zs), program, prompt)
          delta = delta + next_delta
        program += delta
  ```
  ^ use the above to generate a bunch of full rollouts (using top-p or top-k),
  then evaluate them using an offline approximator for the value function (3 outputs)

- [ ] use sampled rollouts to generate training data for value fn
- [ ] implement fns to compute value fn outputs offline 
- [ ] train/test value function
- [ ] fix use of LINE_START/LINE_END tokens