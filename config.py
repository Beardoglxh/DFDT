predefined_parameters = {
    ("halfcheetah", "medium"): {
        "eta": 5.0,
        "grad_norm": 15.0,
        "action_num": 20,
        "dynamics_times": 18,
        "max_uncertainty": 30
    },
    ("halfcheetah", "medium-replay"): {
        "eta": 5.0,
        "grad_norm": 15.0,
        "action_num": 20,
        "dynamics_times": 18,
        "max_uncertainty": 30
    },
    ("halfcheetah", "medium-expert"): {
        "eta": 2.5,
        "grad_norm": 15.0,
        "action_num": 20,
        "dynamics_times": 18,
        "max_uncertainty": 30
    },

    ("hopper", "medium"): {
        "eta": 1.0,
        "grad_norm": 9.0,
        "action_num": 3,
        "dynamics_times": 12,
        "max_uncertainty": 1.0
    },
    ("hopper", "medium-replay"): {
        "eta": 3.0,
        "grad_norm": 9.0,
        "action_num": 3,
        "dynamics_times": 12,
        "max_uncertainty": 1.0
    },
    ("hopper", "medium-expert"): {
        "eta": 1.0,
        "grad_norm": 9.0,
        "action_num": 3,
        "dynamics_times": 12,
        "max_uncertainty": 1.0
    },

    ("walker2d", "medium"): {
        "eta": 2.0,
        "grad_norm": 5.0,
        "action_num": 3,
        "dynamics_times": 18,
        "max_uncertainty": 8.0
    },
    ("walker2d", "medium-replay"): {
        "eta": 2.0,
        "grad_norm": 5.0,
        "action_num": 3,
        "dynamics_times": 18,
        "max_uncertainty": 8.0
    },
    ("walker2d", "medium-expert"): {
        "eta": 2.0,
        "grad_norm": 5.0,
        "action_num": 3,
        "dynamics_times": 18,
        "max_uncertainty": 8.0
    },
    ("pen", "human"): {
        "eta": 0.1,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 46,
        "max_uncertainty": 8.0
    },
    ("hammer", "human"): {
        "eta": 0.1,
        "grad_norm": 5.0,
        "action_num": 1,
        "dynamics_times": 47,
        "max_uncertainty": 1e-10
    },
    ("door", "human"): {
        "eta": 0.005,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 40,
        "max_uncertainty": 1e-10
    },
    ("pen", "cloned"): {
        "eta": 0.01,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 46,
        "max_uncertainty": 1e-10
    },
    ("hammer", "cloned"): {
        "eta": 0.01,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 47,
        "max_uncertainty": 1e-10
    },
    ("door", "cloned"): {
        "eta": 0.001,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 40,
        "max_uncertainty": 1e-10
    },
    ("kitchen", "complete"): {
        "eta": 0.005,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 61,
        "max_uncertainty": 1e-10
    },
    ("kitchen", "partial"): {
        "eta": 0.01,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 61,
        "max_uncertainty": 1e-10
    },
    ("maze2d", "umaze"): {
        "eta": 5.0,
        "grad_norm": 90.0,
        "action_num": 1,
        "dynamics_times": 5,
        "max_uncertainty": 1e-10
    },
    ("maze2d", "medium"): {
        "eta": 5.0,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 5,
        "max_uncertainty": 1e-10
    },
    ("maze2d", "large"): {
        "eta": 4.0,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 5,
        "max_uncertainty": 1e-10
    },
    ("antmaze", "umaze"): {
        "eta": 0.05,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 30,
        "max_uncertainty": 1e-10
    },
    ("antmaze", "umaze-diverse"): {
        "eta": 0.01,
        "grad_norm": 9.0,
        "action_num": 1,
        "dynamics_times": 30,
        "max_uncertainty": 1e-10
    },
}
