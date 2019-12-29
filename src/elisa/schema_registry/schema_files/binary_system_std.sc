{
  "type": "object",
  "required": [
    "system",
    "primary",
    "secondary"
  ],
  "properties": {
    "system": {
      "type": "object",
      "required": [
        "inclination",
        "period",
        "argument_of_periastron",
        "gamma",
        "eccentricity"
      ],
      "properties": {
        "inclination": {
          "type": "number"
        },
        "period": {
          "type": "number"
        },
        "argument_of_periastron": {
          "type": "number"
        },
        "gamma": {
          "type": "number"
        },
        "eccentricity": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "primary_minimum_time": {
          "type": "number"
        },
        "phase_shift": {
          "type": "number"
        }
      }
    },
    "primary": {
      "type": "object",
      "required": [
        "mass",
        "surface_potential",
        "synchronicity",
        "t_eff",
        "gravity_darkening",
        "albedo",
        "metallicity"
      ],
      "properties": {
        "mass": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 100.0
        },
        "surface_potential": {
          "type": "number",
          "minimum": 0.0
        },
        "synchronicity": {
          "type": "number",
          "minimum": 0.0
        },
        "t_eff": {
          "type": "number",
          "minimum": 3500.0,
          "maximum": 50000.0
        },
        "gravity_darkening": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "discretization_factor": {
          "type": "number",
          "minimum": 0.0
        },
        "albedo": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "metallicity": {
          "type": "number"
        }
      }
    }
  },
  "secondary": {
    "type": "object",
    "required": [
      "mass",
      "surface_potential",
      "synchronicity",
      "t_eff",
      "gravity_darkening",
      "albedo",
      "metallicity"
    ],
    "properties": {
      "mass": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 100.0
      },
      "surface_potential": {
        "type": "number",
        "minimum": 0.0
      },
      "synchronicity": {
        "type": "number",
        "minimum": 0.0
      },
      "t_eff": {
        "type": "number",
        "minimum": 3500.0,
        "maximum": 50000.0
      },
      "gravity_darkening": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0
      },
      "discretization_factor": {
        "type": "number",
        "minimum": 0.0
      },
      "albedo": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0
      },
      "metallicity": {
        "type": "number"
      }
    }
  }
}
