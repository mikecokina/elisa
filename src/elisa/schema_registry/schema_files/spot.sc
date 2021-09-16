{
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "longitude",
      "latitude",
      "angular_radius",
      "temperature_factor"
    ],
    "properties": {
      "longitude": {"anyOf": [
        {"type": "number"},
        {"type": "string"}
      ]},
      "latitude": {"anyOf": [
        {"type": "number", "minimum": 0.0, "maximum": 180.0},
        {"type": "string"}
      ]},
      "angular_radius": {"anyOf": [
        {"type": "number", "minimum": 0.0},
        {"type": "string"}
      ]},
      "temperature_factor": {
        "type": "number",
        "minimum": 0
      },
      "discretization_factor": {
        "type": "number",
        "minimum": 0
      }
    }
  }
}