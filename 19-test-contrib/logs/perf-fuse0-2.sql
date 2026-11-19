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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:33Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,1,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '1.125174', '0.000000', '40.687112', '48', '892819', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:34Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '1.639162', '0.000000', '18.624652', '32', '614250', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:35Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '0.321680', '0.000000', '35.608047', '12', '3111440', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:36Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,512,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '144.635742', '0.000000', '108.346499', '16400', '7168', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:38Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '571.919922', '0.000000', '27.430315', '16386', '2048', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:39Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '40.816117', '0.000000', '143.859265', '6148', '24570', '', '');
