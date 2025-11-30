from torchviz import make_dot
dot = make_dot(features['p2'], params=dict(self.named_parameters()))
dot.render("computation_graph", format="png")

from torchviz import make_dot
dot = make_dot(features['p2'], params=dict(self.named_parameters()))
dot.render("computation_graph", format="png")