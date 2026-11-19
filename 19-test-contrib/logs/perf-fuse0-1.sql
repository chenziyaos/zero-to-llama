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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:20Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,1,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '0.911672', '0.000000', '50.215516', '48', '1097594', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:21Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '2.362781', '0.000000', '12.920715', '32', '425880', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:22Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,1,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '0.846269', '0.000000', '13.535176', '12', '1187260', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:23Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[4096,512,1,1],eps=0.000001,multi_add=0,mul=1,rope=0,set_rows=0,broadcast=1,mode=0,set_rows_type=f16', 'perf', '1', '1', '', '144.858677', '0.000000', '108.179756', '16400', '7168', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:24Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,32,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=0,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '587.271484', '0.000000', '26.713273', '16386', '2048', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-11-01T16:37:25Z', '0ec4027', 'CPU', 'RMS_NORM_MUL_ROPE', 'ne=[128,8,512,1],eps=0.000001,multi_add=0,mul=1,rope=1,set_rows=1,broadcast=1,mode=2,set_rows_type=f16', 'perf', '1', '1', '', '41.197436', '0.000000', '142.527720', '6148', '24570', '', '');
