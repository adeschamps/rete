# Rete

This is a Rust implementation of the [Rete pattern matching
algorithm][wiki]. It is based largely on the paper [_Production
Matching for Large Learning Systems_][Doorenbos] by Robert Doorenbos, but adapted to be more idiomatic Rust.

It is currently incomplete. Basic features are in progress:

- [x] Production addition
- [ ] Production removal
- [x] Wme addition
- [ ] Wme removal

Additionally, it only handles equality checks (no numeric comparisons) and it does not yet handle negated conditions.

[wiki]: https://en.wikipedia.org/wiki/Rete_algorithm
[Doorenbos]: http://reports-archive.adm.cs.cmu.edu/anon/1995/CMU-CS-95-113.pdf