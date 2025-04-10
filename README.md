# MapColouringUsingLocalBeamSearch
The attached script uses Local Beam Search for map colouring. For this purpose we take care
of constraints,degree-based coloring, pre-assigned colors, balanced color usage, and distance
constraints.
A state representation is defined, where each state consists of a coloring assignment,
conflict count, color usage, and a balance penalty. The heuristic function is designed to
minimize conflicts and reduce the number of colors used.
The initial state is generated by sorting vertices based on degree and assigning colors
sequentially while ensuring no adjacent or two-hop neighbor conflicts occur.
Pre-assigned colors are preserved.
# Execution
 The Local Beam Search algorithm is then executed, maintaining a fixed number of best
states and iteratively refining them by selecting the lowest-conflict states. The search
stops when either a zero-conflict coloring is found or the maximum iteration limit is
reached.
