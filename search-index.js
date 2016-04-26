var searchIndex = {};
searchIndex["vectormath"] = {"doc":"","items":[[5,"min","vectormath","",null,{"inputs":[{"name":"t"},{"name":"t"}],"output":{"name":"t"}}],[5,"max","","",null,{"inputs":[{"name":"t"},{"name":"t"}],"output":{"name":"t"}}],[0,"float","","Support module for floating-point types",null,null],[8,"Float","vectormath::float","Floating-point type trait",null,null],[10,"one","","Constant 1",0,{"inputs":[],"output":{"name":"self"}}],[10,"zero","","Constant 0",0,{"inputs":[],"output":{"name":"self"}}],[10,"two","","Constant 2",0,{"inputs":[],"output":{"name":"self"}}],[10,"onehalf","","Constant 0.5",0,{"inputs":[],"output":{"name":"self"}}],[10,"pi","","Constant PI",0,{"inputs":[],"output":{"name":"self"}}],[10,"pi_by_c180","","Constant PI / 180",0,{"inputs":[],"output":{"name":"self"}}],[10,"c180_by_pi","","Constant 180 / PI",0,{"inputs":[],"output":{"name":"self"}}],[10,"epsilon","","Constant EPSILON",0,{"inputs":[],"output":{"name":"self"}}],[8,"Cmp","","Comparison",null,null],[16,"Bool","","Corresponding boolean type",1,null],[10,"eq","","Test equality",1,{"inputs":[{"name":"cmp"},{"name":"self"}],"output":{"name":"bool"}}],[10,"ne","","Test inequality",1,{"inputs":[{"name":"cmp"},{"name":"self"}],"output":{"name":"bool"}}],[10,"gt","","Test greater than",1,{"inputs":[{"name":"cmp"},{"name":"self"}],"output":{"name":"bool"}}],[10,"lt","","Test less than",1,{"inputs":[{"name":"cmp"},{"name":"self"}],"output":{"name":"bool"}}],[10,"ge","","Test greater than or equal to",1,{"inputs":[{"name":"cmp"},{"name":"self"}],"output":{"name":"bool"}}],[10,"le","","Test less than or equal to",1,{"inputs":[{"name":"cmp"},{"name":"self"}],"output":{"name":"bool"}}],[8,"Sel","","Selection",null,null],[10,"sel","","Element-wise selection\nResult[i] = (rhs[i] if cond[i] == true, self[i] otherwise)",2,{"inputs":[{"name":"sel"},{"name":"self"},{"name":"bool"}],"output":{"name":"self"}}],[8,"Ops","","Trait of Operations",null,null],[10,"abs","","Absolute Value",3,{"inputs":[{"name":"ops"}],"output":{"name":"self"}}],[10,"recip","","Reciprocal Value",3,{"inputs":[{"name":"ops"}],"output":{"name":"self"}}],[10,"sqrt","","Square Root",3,{"inputs":[{"name":"ops"}],"output":{"name":"self"}}],[10,"rsqrt","","Reciprocal Square Root",3,{"inputs":[{"name":"ops"}],"output":{"name":"self"}}],[8,"Trig","","Trait of Trigonometry",null,null],[10,"sin","","Sine",4,{"inputs":[{"name":"trig"}],"output":{"name":"self"}}],[10,"cos","","Cosine",4,{"inputs":[{"name":"trig"}],"output":{"name":"self"}}],[10,"sincos","","Returns (sin(x), cos(x))",4,null],[10,"tan","","Tangent",4,{"inputs":[{"name":"trig"}],"output":{"name":"self"}}],[10,"acos","","Arccosine",4,{"inputs":[{"name":"trig"}],"output":{"name":"self"}}],[8,"Min","","Trait of Minimum",null,null],[10,"min","","Returns the minimum between values",5,{"inputs":[{"name":"min"},{"name":"self"}],"output":{"name":"self"}}],[8,"Max","","Trait of Maximum",null,null],[10,"max","","Returns the maximum between values",6,{"inputs":[{"name":"max"},{"name":"self"}],"output":{"name":"self"}}],[8,"Clamp","","Trait of Clamp",null,null],[10,"clamp","","Constrain a value to lie between two further values",7,{"inputs":[{"name":"clamp"},{"name":"self"},{"name":"self"}],"output":{"name":"self"}}],[0,"angle","vectormath","Radian and Degree",null,null],[3,"Rad","vectormath::angle","Angle in radians",null,null],[12,"0","","",8,null],[3,"Deg","","Angle in degrees",null,null],[12,"0","","",9,null],[11,"clone","","",8,{"inputs":[{"name":"rad"}],"output":{"name":"rad"}}],[11,"fmt","","",8,{"inputs":[{"name":"rad"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",9,{"inputs":[{"name":"deg"}],"output":{"name":"deg"}}],[11,"fmt","","",9,{"inputs":[{"name":"deg"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"from","","",9,{"inputs":[{"name":"rad"}],"output":{"name":"deg"}}],[11,"from","","",8,{"inputs":[{"name":"deg"}],"output":{"name":"rad"}}],[0,"vector","vectormath","Vector",null,null],[3,"Vec3","vectormath::vector","3-D Vector",null,null],[12,"x","","1st component",10,null],[12,"y","","2nd component",10,null],[12,"z","","3rd component",10,null],[3,"Pos3","","3-D Position",null,null],[12,"x","","1st component",11,null],[12,"y","","2nd component",11,null],[12,"z","","3rd component",11,null],[3,"Vec4","","4-D Vector",null,null],[12,"x","","1st component",12,null],[12,"y","","2nd component",12,null],[12,"z","","3rd component",12,null],[12,"w","","4th component",12,null],[11,"clone","","",10,{"inputs":[{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"fmt","","",10,{"inputs":[{"name":"vec3"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",11,{"inputs":[{"name":"pos3"}],"output":{"name":"pos3"}}],[11,"fmt","","",11,{"inputs":[{"name":"pos3"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",12,{"inputs":[{"name":"vec4"}],"output":{"name":"vec4"}}],[11,"fmt","","",12,{"inputs":[{"name":"vec4"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"add","","",10,{"inputs":[{"name":"vec3"},{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"add","","",12,{"inputs":[{"name":"vec4"},{"name":"vec4"}],"output":{"name":"vec4"}}],[11,"add","","",11,{"inputs":[{"name":"pos3"},{"name":"vec3"}],"output":{"name":"pos3"}}],[11,"sub","","",10,{"inputs":[{"name":"vec3"},{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"sub","","",12,{"inputs":[{"name":"vec4"},{"name":"vec4"}],"output":{"name":"vec4"}}],[11,"sub","","",11,{"inputs":[{"name":"pos3"},{"name":"vec3"}],"output":{"name":"pos3"}}],[11,"sub","","",11,{"inputs":[{"name":"pos3"},{"name":"pos3"}],"output":{"name":"vec3"}}],[11,"mul","","",10,{"inputs":[{"name":"vec3"},{"name":"t"}],"output":{"name":"vec3"}}],[11,"mul","","",12,{"inputs":[{"name":"vec4"},{"name":"t"}],"output":{"name":"vec4"}}],[11,"mul","","",11,{"inputs":[{"name":"pos3"},{"name":"t"}],"output":{"name":"pos3"}}],[11,"div","","",10,{"inputs":[{"name":"vec3"},{"name":"t"}],"output":{"name":"vec3"}}],[11,"div","","",12,{"inputs":[{"name":"vec4"},{"name":"t"}],"output":{"name":"vec4"}}],[11,"div","","",11,{"inputs":[{"name":"pos3"},{"name":"t"}],"output":{"name":"pos3"}}],[11,"neg","","",10,{"inputs":[{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"neg","","",12,{"inputs":[{"name":"vec4"}],"output":{"name":"vec4"}}],[11,"neg","","",11,{"inputs":[{"name":"pos3"}],"output":{"name":"pos3"}}],[11,"dot","","Dot Product",10,{"inputs":[{"name":"vec3"},{"name":"self"}],"output":{"name":"t"}}],[11,"cross","","Cross Product",10,{"inputs":[{"name":"vec3"},{"name":"self"}],"output":{"name":"self"}}],[11,"length","","Magnitude",10,{"inputs":[{"name":"vec3"}],"output":{"name":"t"}}],[11,"length_squared","","Squared Magnitude",10,{"inputs":[{"name":"vec3"}],"output":{"name":"t"}}],[11,"normalize","","Normalization",10,{"inputs":[{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"lerp","","Linear Interpolation",10,{"inputs":[{"name":"vec3"},{"name":"vec3"},{"name":"t"}],"output":{"name":"vec3"}}],[11,"dot","","Dot Product",12,{"inputs":[{"name":"vec4"},{"name":"self"}],"output":{"name":"t"}}],[11,"length","","Magnitude",12,{"inputs":[{"name":"vec4"}],"output":{"name":"t"}}],[11,"length_squared","","Squared Magnitude",12,{"inputs":[{"name":"vec4"}],"output":{"name":"t"}}],[11,"normalize","","Normalization",12,{"inputs":[{"name":"vec4"}],"output":{"name":"vec4"}}],[11,"lerp","","Linear Interpolation",12,{"inputs":[{"name":"vec4"},{"name":"vec4"},{"name":"t"}],"output":{"name":"vec4"}}],[11,"lerp","","Linear Interpolation",11,{"inputs":[{"name":"pos3"},{"name":"pos3"},{"name":"t"}],"output":{"name":"pos3"}}],[11,"min","","",10,{"inputs":[{"name":"vec3"},{"name":"self"}],"output":{"name":"self"}}],[11,"min","","",12,{"inputs":[{"name":"vec4"},{"name":"self"}],"output":{"name":"self"}}],[11,"min","","",11,{"inputs":[{"name":"pos3"},{"name":"self"}],"output":{"name":"self"}}],[11,"max","","",10,{"inputs":[{"name":"vec3"},{"name":"self"}],"output":{"name":"self"}}],[11,"max","","",12,{"inputs":[{"name":"vec4"},{"name":"self"}],"output":{"name":"self"}}],[11,"max","","",11,{"inputs":[{"name":"pos3"},{"name":"self"}],"output":{"name":"self"}}],[11,"from","","",10,{"inputs":[{"name":"pos3"}],"output":{"name":"vec3"}}],[11,"from","","",10,{"inputs":[{"name":"vec4"}],"output":{"name":"vec3"}}],[11,"from","","",12,{"inputs":[{"name":"vec3"}],"output":{"name":"vec4"}}],[11,"from","","",12,{"inputs":[{"name":"pos3"}],"output":{"name":"vec4"}}],[11,"from","","",11,{"inputs":[{"name":"vec3"}],"output":{"name":"pos3"}}],[11,"from","","",11,{"inputs":[{"name":"vec4"}],"output":{"name":"pos3"}}],[11,"x_axis","","X Axis",10,{"inputs":[],"output":{"name":"vec3"}}],[11,"y_axis","","Y Axis",10,{"inputs":[],"output":{"name":"vec3"}}],[11,"z_axis","","Z Axis",10,{"inputs":[],"output":{"name":"vec3"}}],[11,"x_axis","","X Axis",12,{"inputs":[],"output":{"name":"vec4"}}],[11,"y_axis","","Y Axis",12,{"inputs":[],"output":{"name":"vec4"}}],[11,"z_axis","","Z Axis",12,{"inputs":[],"output":{"name":"vec4"}}],[11,"w_axis","","W Axis",12,{"inputs":[],"output":{"name":"vec4"}}],[11,"x_axis","","X Axis",11,{"inputs":[],"output":{"name":"pos3"}}],[11,"y_axis","","Y Axis",11,{"inputs":[],"output":{"name":"pos3"}}],[11,"z_axis","","Z Axis",11,{"inputs":[],"output":{"name":"pos3"}}],[0,"matrix","vectormath","Matrix",null,null],[3,"Mat3","vectormath::matrix","3x3 Column-major Matrix",null,null],[12,"0","","",13,null],[12,"1","","",13,null],[12,"2","","",13,null],[3,"Mat4","","4x4 Column-major Matrix",null,null],[12,"0","","",14,null],[12,"1","","",14,null],[12,"2","","",14,null],[12,"3","","",14,null],[3,"Tfm3","","3-D Affine Transform Matrix",null,null],[12,"0","","",15,null],[12,"1","","",15,null],[12,"2","","",15,null],[12,"3","","",15,null],[11,"clone","","",13,{"inputs":[{"name":"mat3"}],"output":{"name":"mat3"}}],[11,"fmt","","",13,{"inputs":[{"name":"mat3"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",14,{"inputs":[{"name":"mat4"}],"output":{"name":"mat4"}}],[11,"fmt","","",14,{"inputs":[{"name":"mat4"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",15,{"inputs":[{"name":"tfm3"}],"output":{"name":"tfm3"}}],[11,"fmt","","",15,{"inputs":[{"name":"tfm3"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"add","","",13,{"inputs":[{"name":"mat3"},{"name":"mat3"}],"output":{"name":"mat3"}}],[11,"add","","",14,{"inputs":[{"name":"mat4"},{"name":"mat4"}],"output":{"name":"mat4"}}],[11,"add","","",15,{"inputs":[{"name":"tfm3"},{"name":"tfm3"}],"output":{"name":"tfm3"}}],[11,"sub","","",13,{"inputs":[{"name":"mat3"},{"name":"mat3"}],"output":{"name":"mat3"}}],[11,"sub","","",14,{"inputs":[{"name":"mat4"},{"name":"mat4"}],"output":{"name":"mat4"}}],[11,"sub","","",15,{"inputs":[{"name":"tfm3"},{"name":"tfm3"}],"output":{"name":"tfm3"}}],[11,"mul","","",13,{"inputs":[{"name":"mat3"},{"name":"t"}],"output":{"name":"mat3"}}],[11,"mul","","",13,{"inputs":[{"name":"mat3"},{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"mul","","",13,{"inputs":[{"name":"mat3"},{"name":"pos3"}],"output":{"name":"pos3"}}],[11,"mul","","",13,{"inputs":[{"name":"mat3"},{"name":"mat3"}],"output":{"name":"mat3"}}],[11,"mul","","",14,{"inputs":[{"name":"mat4"},{"name":"t"}],"output":{"name":"mat4"}}],[11,"mul","","",14,{"inputs":[{"name":"mat4"},{"name":"vec4"}],"output":{"name":"vec4"}}],[11,"mul","","",14,{"inputs":[{"name":"mat4"},{"name":"pos3"}],"output":{"name":"vec4"}}],[11,"mul","","",14,{"inputs":[{"name":"mat4"},{"name":"mat4"}],"output":{"name":"mat4"}}],[11,"mul","","",15,{"inputs":[{"name":"tfm3"},{"name":"t"}],"output":{"name":"tfm3"}}],[11,"mul","","",15,{"inputs":[{"name":"tfm3"},{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"mul","","",15,{"inputs":[{"name":"tfm3"},{"name":"vec4"}],"output":{"name":"vec3"}}],[11,"mul","","",15,{"inputs":[{"name":"tfm3"},{"name":"pos3"}],"output":{"name":"pos3"}}],[11,"mul","","",15,{"inputs":[{"name":"tfm3"},{"name":"tfm3"}],"output":{"name":"tfm3"}}],[11,"neg","","",13,{"inputs":[{"name":"mat3"}],"output":{"name":"mat3"}}],[11,"neg","","",14,{"inputs":[{"name":"mat4"}],"output":{"name":"mat4"}}],[11,"neg","","",15,{"inputs":[{"name":"tfm3"}],"output":{"name":"tfm3"}}],[11,"transpose","","Transposition",13,{"inputs":[{"name":"mat3"}],"output":{"name":"mat3"}}],[11,"transpose","","Transposition",14,{"inputs":[{"name":"mat4"}],"output":{"name":"mat4"}}],[11,"inverse","","Inverse of the matrix",13,{"inputs":[{"name":"mat3"}],"output":{"name":"mat3"}}],[11,"inverse","","Inverse of the matrix",14,{"inputs":[{"name":"mat4"}],"output":{"name":"mat4"}}],[11,"inverse","","Inversion",15,{"inputs":[{"name":"tfm3"}],"output":{"name":"tfm3"}}],[11,"from","","",13,{"inputs":[{"name":"mat4"}],"output":{"name":"mat3"}}],[11,"from","","",13,{"inputs":[{"name":"tfm3"}],"output":{"name":"mat3"}}],[11,"from","","",14,{"inputs":[{"name":"mat3"}],"output":{"name":"mat4"}}],[11,"from","","",14,{"inputs":[{"name":"tfm3"}],"output":{"name":"mat4"}}],[11,"from","","",15,{"inputs":[{"name":"mat3"}],"output":{"name":"tfm3"}}],[11,"identity","","Construct an identity matrix",13,{"inputs":[],"output":{"name":"mat3"}}],[11,"rotation_angle_axis","","Construct a rotation matrix",13,{"inputs":[{"name":"rad"},{"name":"vec3"}],"output":{"name":"mat3"}}],[11,"identity","","Construct an identity matrix",14,{"inputs":[],"output":{"name":"mat4"}}],[11,"identity","","Construct an identity matrix",15,{"inputs":[],"output":{"name":"tfm3"}}],[0,"quaternion","vectormath","Quaternion",null,null],[3,"Quat","vectormath::quaternion","Quaternion",null,null],[12,"w","","real part (scalar part)",16,null],[12,"x","","1st imaginary part (x of vector part)",16,null],[12,"y","","2nd imaginary part (y of vector part)",16,null],[12,"z","","3rd imaginary part (z of vector part)",16,null],[11,"clone","","",16,{"inputs":[{"name":"quat"}],"output":{"name":"quat"}}],[11,"fmt","","",16,{"inputs":[{"name":"quat"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"add","","",16,{"inputs":[{"name":"quat"},{"name":"quat"}],"output":{"name":"quat"}}],[11,"sub","","",16,{"inputs":[{"name":"quat"},{"name":"quat"}],"output":{"name":"quat"}}],[11,"mul","","",16,{"inputs":[{"name":"quat"},{"name":"t"}],"output":{"name":"quat"}}],[11,"mul","","",16,{"inputs":[{"name":"quat"},{"name":"quat"}],"output":{"name":"quat"}}],[11,"neg","","",16,{"inputs":[{"name":"quat"}],"output":{"name":"quat"}}],[11,"conj","","Conjugate",16,{"inputs":[{"name":"quat"}],"output":{"name":"quat"}}],[11,"norm","","Magnitude",16,{"inputs":[{"name":"quat"}],"output":{"name":"t"}}],[11,"norm_squared","","Squared Magnitude",16,{"inputs":[{"name":"quat"}],"output":{"name":"t"}}],[11,"normalize","","Normalize",16,{"inputs":[{"name":"quat"}],"output":{"name":"quat"}}],[11,"dot","","Dot Product",16,{"inputs":[{"name":"quat"},{"name":"quat"}],"output":{"name":"t"}}],[11,"rotate","","Transform a vector with the quaternion",16,{"inputs":[{"name":"quat"},{"name":"vec3"}],"output":{"name":"vec3"}}],[11,"lerp","","linear interpolation",16,{"inputs":[{"name":"quat"},{"name":"quat"},{"name":"t"}],"output":{"name":"quat"}}],[11,"slerp","","spherical linear interpolation",16,{"inputs":[{"name":"quat"},{"name":"quat"},{"name":"t"}],"output":{"name":"quat"}}],[11,"xyz","","extract xyz components",16,{"inputs":[{"name":"quat"}],"output":{"name":"vec3"}}],[11,"from","","",16,{"inputs":[{"name":"mat3"}],"output":{"name":"quat"}}],[11,"from","vectormath::matrix","",13,{"inputs":[{"name":"quat"}],"output":{"name":"mat3"}}],[11,"from","vectormath::quaternion","",16,{"inputs":[{"name":"vec3"}],"output":{"name":"quat"}}],[11,"identity","","Construct an identity quaternion",16,{"inputs":[],"output":{"name":"quat"}}],[11,"zero","","Construct a zero quaternion",16,{"inputs":[],"output":{"name":"quat"}}],[11,"from_angle_axis","","Construct a quaternion that rotates by `angle` around `axis`",16,{"inputs":[{"name":"rad"},{"name":"vec3"}],"output":{"name":"quat"}}],[11,"from_vectors","","Construct a quaternion that rotates from one vector to another",16,{"inputs":[{"name":"vec3"},{"name":"vec3"}],"output":{"name":"quat"}}],[0,"plane","vectormath","Plane",null,null],[3,"Plane","vectormath::plane","Plane",null,null],[12,"normal","","Normal vector (`a`, `b`, `c` coefficients)",17,null],[12,"d","","`d` coefficient",17,null],[11,"fmt","","",17,{"inputs":[{"name":"plane"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",17,{"inputs":[{"name":"plane"}],"output":{"name":"plane"}}],[11,"distance","","Signed distance to a point",17,{"inputs":[{"name":"plane"},{"name":"pos3"}],"output":{"name":"t"}}],[14,"assert_approx_eq!","vectormath","",null,null]],"paths":[[8,"Float"],[8,"Cmp"],[8,"Sel"],[8,"Ops"],[8,"Trig"],[8,"Min"],[8,"Max"],[8,"Clamp"],[3,"Rad"],[3,"Deg"],[3,"Vec3"],[3,"Pos3"],[3,"Vec4"],[3,"Mat3"],[3,"Mat4"],[3,"Tfm3"],[3,"Quat"],[3,"Plane"]]};
initSearch(searchIndex);