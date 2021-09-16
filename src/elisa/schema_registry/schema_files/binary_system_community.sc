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
        "eccentricity",
        "semi_major_axis",
        "mass_ratio"
      ],
      "properties": {
        "inclination": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "period": {"anyOf": [
          {"type": "number", "minimum": 0.0},
          {"type": "string"}
        ]},
        "argument_of_periastron": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "gamma": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "eccentricity": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "semi_major_axis": {"anyOf": [
          {"type": "number", "minimum": 0.0},
          {"type": "string"}
        ]},
        "mass_ratio": {
          "type": "number",
          "minimum": 0.0
        },
        "primary_minimum_time": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
        "phase_shift": {
          "type": "number"
        },
        "additional_light": {
          "type": "number"
        }
      }
    },
    "primary": {
      "type": "object",
      "required": [
        "surface_potential",
        "synchronicity",
        "t_eff"
      ],
      "properties": {
        "surface_potential": {
          "type": "number",
          "minimum": 0.0
        },
        "synchronicity": {
          "type": "number",
          "minimum": 0.0
        },
        "t_eff": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
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
        },
        "atmosphere": {
          "type": "string"
        }
      }
    },
    "secondary": {
      "type": "object",
      "required": [
        "surface_potential",
        "synchronicity",
        "t_eff"
      ],
      "properties": {
        "surface_potential": {
          "type": "number",
          "minimum": 0.0
        },
        "synchronicity": {
          "type": "number",
          "minimum": 0.0
        },
        "t_eff": {"anyOf": [
          {"type": "number"},
          {"type": "string"}
        ]},
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
        },
        "atmosphere": {
          "type": "string"
        }
      }
    }
  }
}
