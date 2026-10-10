import gaiya
from gaiya import Plane, Property

# the ARC field: a flow plane wrapping r, c, cell varieties
arc_field=Plane(
    grid=lambda g: (len(g), len(g[0])),
    properties=[
        Property("colour",
                needs={"subject", "position"},
                measure=lambda ctx: ctx.subject[ctx.position[0]][ctx.position[1]],
                ),
    ],
)
