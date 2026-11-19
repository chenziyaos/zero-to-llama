CREATE TABLE IF NOT EXISTS test_backend_ops (
  test_time TEXT,
  build_commit TEXT,
  backend_name TEXT,
  op_name TEXT,
  op_params TEXT,
  test_mode TEXT,
  supported INTEGER,
  passed INTEGER,
  error_message TEXT,
  time_us REAL,
  flops REAL,
  bandwidth_gb_s REAL,
  memory_kb INTEGER,
  n_runs INTEGER,
  device_description TEXT,
  backend_reg_name TEXT
);

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:13Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,1,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '0.753843', '0.000000', '60.728945', '48', '1326942', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:14Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '1.583341', '0.000000', '19.281261', '32', '638820', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:15Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '0.316102', '0.000000', '36.236433', '12', '3168756', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:16Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,512,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '141.750279', '0.000000', '110.551997', '16400', '7168', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:17Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '581.637695', '0.000000', '26.972020', '16386', '2048', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:18Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '44.903426', '0.000000', '130.764557', '6148', '23205', '', '');
