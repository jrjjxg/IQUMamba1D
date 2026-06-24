function GEN_8PSK_MultiSource(varargin)
%GEN_8PSK_MultiSource  Generate strict Table(a) 8PSK-A/B dataset (Ideal AWGN).
%
% Default behavior:
%   cd('E:\MT-RF\IQUMamba1D\dataset_gen');
%   GEN_8PSK_MultiSource();              % 10 files per SNR, SNR=-10:4:30
%   GEN_8PSK_MultiSource(10);            % set files per SNR
%   GEN_8PSK_MultiSource(10, -10:4:30);  % set files and SNR list
%   GEN_8PSK_MultiSource(10, -10:4:30, 'bb'); % SNR defined on saved baseband frames (default)
%   GEN_8PSK_MultiSource(10, -10:4:30, 'rf'); % SNR defined on r(t) before downconvert/LPF
%   GEN_8PSK_MultiSource('8PSK-B');      % Table(a) 8PSK-B, 10 files per SNR
%   GEN_8PSK_MultiSource('8PSK-B', 10);  % Table(a) 8PSK-B, set files per SNR
%
% Legacy behavior:
%   GEN_8PSK_MultiSource('legacy');      % uses the original defaults

    if nargin >= 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'legacy')
        for snr = -10:4:30
            GEN_8PSK_MultiSource_i(2, snr, 1);
        end
        return;
    end

    dataset_name = '8PSK-A';
    arg_offset = 0;
    if nargin >= 1 && (ischar(varargin{1}) || isstring(varargin{1}))
        dataset_name = char(varargin{1});
        arg_offset = 1;
    end
    dataset_name = normalize_tablea_8psk_dataset(dataset_name);

    num_files = 10;
    snr_list = -10:4:30;
    snr_domain = 'bb';
    if nargin >= arg_offset + 1 && ~isempty(varargin{arg_offset + 1})
        num_files = varargin{arg_offset + 1};
    end
    if nargin >= arg_offset + 2 && ~isempty(varargin{arg_offset + 2})
        snr_list = varargin{arg_offset + 2};
    end
    if nargin >= arg_offset + 3 && ~isempty(varargin{arg_offset + 3})
        snr_domain = varargin{arg_offset + 3};
    end
    snr_domain = normalize_snr_domain(snr_domain);

    for snr = snr_list
        GEN_8PSK_MultiSource_i(2, snr, num_files, false, [], dataset_name, true, snr_domain);
    end
end

function GEN_8PSK_MultiSource_i(num_sources, snr, num_files, impaired, output_root, dataset_name, strict_tablea_8psk_a, snr_domain)
    % Multi-source 8PSK signal generation function
    % num_sources: Number of sources (2, 3, 4)
    % snr: Signal-to-noise ratio (dB)
    % num_files: Number of files
    
    %% Parameter validation
    if nargin < 1
        num_sources = 2;  % Default dual-source
    end
    if nargin < 2
        snr = 25;  % Default 25dB
    end
    if nargin < 3
        num_files = 10; % Default: 10 files per SNR (paper setting)
    end
    if nargin < 4
        impaired = true; % Default: enable non-ideal characteristics
    end
    if nargin < 6 || isempty(dataset_name)
        dataset_name = '8PSK';
    end
    if nargin < 7 || isempty(strict_tablea_8psk_a)
        strict_tablea_8psk_a = false;
    end
    if nargin < 8 || isempty(snr_domain)
        snr_domain = 'bb';
    end
    snr_domain = normalize_snr_domain(snr_domain);

    dataset_case = upper(strtrim(dataset_name));
    if strict_tablea_8psk_a
        dataset_case = normalize_tablea_8psk_dataset(dataset_case);
        dataset_name = dataset_case;
        num_sources = 2;
        impaired = false;
    end
    
    if ~ismember(num_sources, [2, 3, 4])
        error('Number of sources must be 2, 3, or 4');
    end
    
    %% ========== Added: Non-ideal characteristics parameters ==========
    % 1. Carrier frequency drift parameters
    enable_carrier_drift = impaired;
    carrier_drift_rate = 50;            % Hz/s, carrier frequency drift rate (linear drift)
    carrier_drift_random_walk_std = 5;  % Hz, random walk standard deviation
    carrier_drift_type = 'combined';    % 'linear', 'random_walk', 'sinusoidal', 'combined'
    
    % Sinusoidal frequency modulation parameters (simulate oscillator instability)
    carrier_fm_amplitude = 20;          % Hz, FM amplitude
    carrier_fm_frequency = 2;           % Hz, FM frequency
    
    % 2. Symbol clock jitter parameters
    enable_timing_jitter = impaired;
    timing_jitter_rms = 0.015;          % RMS jitter as fraction of symbol period (1.5%)
    timing_jitter_type = 'gaussian';    % 'gaussian', 'uniform', 'colored'
    
    % 3. Amplitude variation parameters
    enable_amplitude_variation = impaired;
    amplitude_variation_std = 0.03;     % Amplitude variation standard deviation (3%)
    amplitude_variation_bandwidth = 50; % Variation bandwidth (Hz)
    amplitude_fade_depth = 0.1;         % Slow fading depth (10%)
    amplitude_fade_rate = 0.5;          % Slow fading rate (Hz)
    
    if impaired
        fprintf('=== Non-ideal characteristics configuration ===\n');
        fprintf('1. Carrier frequency drift:\n');
        fprintf('   - Drift rate: %d Hz/s\n', carrier_drift_rate);
        fprintf('   - Random walk: std=%d Hz\n', carrier_drift_random_walk_std);
        fprintf('   - FM amplitude: %d Hz @ %d Hz\n', carrier_fm_amplitude, carrier_fm_frequency);
        fprintf('2. Symbol clock jitter:\n');
        fprintf('   - RMS jitter: %.2f%% symbol period\n', timing_jitter_rms*100);
        fprintf('3. Amplitude variation:\n');
        fprintf('   - Fast variation: std=%.1f%%, BW=%d Hz\n', amplitude_variation_std*100, amplitude_variation_bandwidth);
        fprintf('   - Slow fading: depth=%.1f%%, rate=%.2f Hz\n', amplitude_fade_depth*100, amplitude_fade_rate);
        fprintf('\n');
    end
    
    %% Basic parameters
    flo = 20e6;                 % Local oscillator frequency
    Fs_rf = 100e6;              % Sampling rate
    if strict_tablea_8psk_a && strcmp(dataset_case, '8PSK-B')
        flo = 10.000125e6;
        Fs_rf = 50e6;
    end
    
    %% Root-raised cosine filter
    alpha = 0.35;               % Roll-off factor
    span = 20;                   % Filter symbol span
    if strict_tablea_8psk_a && strcmp(dataset_case, '8PSK-B')
        symbol_rates = [2.5e6, 5e6];
        Fs_sps_by_source = [20, 10];
    else
        symbol_rates = 5e6 * ones(1, num_sources);
        Fs_sps_by_source = 20 * ones(1, num_sources);
    end
    Fs_sps = Fs_sps_by_source;  % Saved as metadata; scalar only when every source is identical.
    filterCoeffs = cell(num_sources, 1);
    for i = 1:num_sources
        filterCoeffs{i} = rcosdesign(alpha, span, Fs_sps_by_source(i), 'sqrt');
    end
    
    %% Dataset parameters
    %num_files = 20;              % Number of files
    samples_per_file = 500;     % Number of frames per file
    frame_length = 4096;        % Restore original 8PSK-A frame length for separation experiments
    valid_frame_length = frame_length;
    bits_per_symbol = 3;        % Number of bits per symbol
    total_frames = samples_per_file;
    total_samples = total_frames * frame_length;  % Total sampling points per file
    symbols_per_frame_by_source = round(frame_length ./ Fs_sps_by_source);
    total_symbols_by_source = total_frames .* symbols_per_frame_by_source;
    bits_per_frame_by_source = symbols_per_frame_by_source * bits_per_symbol;
    if all(symbols_per_frame_by_source == symbols_per_frame_by_source(1))
        symbols_per_frame = symbols_per_frame_by_source(1);
        bits_per_frame = bits_per_frame_by_source(1);
    else
        symbols_per_frame = symbols_per_frame_by_source;
        bits_per_frame = bits_per_frame_by_source;
    end
    
    %% Delay parameters (reduce delay to avoid affecting BER)
    if strict_tablea_8psk_a && strcmp(dataset_case, '8PSK-B')
        Tb = 1/min(symbol_rates);
        base_delay = 0.3 * Tb;
    else
        Tb = 1/symbol_rates(1);
        base_delay = 0.05 * Tb;
    end
    delay_samples = round((0:num_sources-1) * base_delay * Fs_rf);
    
    %% Low-pass filter design
    rolloff = 0.35;
    cutoff_freq = max(symbol_rates) * (1+rolloff)/2;
    normalized_cutoff = cutoff_freq/(Fs_rf/2);
    h_lpf_mixed = fir1(127, normalized_cutoff, 'low', kaiser(128, 5));
    h_lpf_sources = cell(num_sources, 1);
    for i = 1:num_sources
        cutoff_freq_i = symbol_rates(i) * (1+rolloff)/2;
        normalized_cutoff_i = cutoff_freq_i/(Fs_rf/2);
        h_lpf_sources{i} = fir1(127, normalized_cutoff_i, 'low', kaiser(128, 5));
    end
    
    %% Constellation mapping table
    constellation = [exp(1j * pi/8);       % dec 0 (000)
                 exp(1j * 3*pi/8);     % dec 1 (001)
                 exp(1j * 5*pi/8);     % dec 3 (011)
                 exp(1j * 7*pi/8);     % dec 2 (010)
                 exp(1j * 9*pi/8);     % dec 6 (110)
                 exp(1j * 11*pi/8);   % dec 7 (111)
                 exp(1j * 13*pi/8);    % dec 5 (101)
                 exp(1j * 15*pi/8)];    % dec 4 (100)   
    % Mapping table: natural binary -> constellation index (Gray code mapping)
    gray_map_array = [1, 2, 4, 3, 8, 7, 5, 6]; % Index mapping [n=0,1,2,3,4,5,6,7]
    
    %% ========== Added: Generate carrier frequency drift function ==========
    function phase_drift = generate_carrier_drift(t, fc_base, drift_params, file_seed)
        % Generate time-varying carrier frequency drift (as phase variation)
        % Input:
        %   t - Time vector
        %   fc_base - Base carrier frequency
        %   drift_params - Drift parameter structure
        % Output:
        %   phase_drift - Phase drift (radians)
        
        rng(file_seed * 7777); % Ensure reproducibility
        Fs = 1/mean(diff(t));
        N = length(t);
        
        phase_drift = zeros(size(t));
        
        % 1. Linear drift component
        if contains(drift_params.type, 'linear') || contains(drift_params.type, 'combined')
            linear_drift = drift_params.rate * t;  % Hz * s = Hz
            phase_drift = phase_drift + 2*pi * cumsum(linear_drift) / Fs;
        end
        
        % 2. Random walk component (simulate slow random drift like temperature changes)
        if contains(drift_params.type, 'random_walk') || contains(drift_params.type, 'combined')
            random_steps = randn(N, 1) * drift_params.random_walk_std;
            % Low-pass filter to make changes smoother
            [b_smooth, a_smooth] = butter(2, 10/(Fs/2));  % 10Hz cutoff
            random_walk = filter(b_smooth, a_smooth, random_steps);
            phase_drift = phase_drift + 2*pi * cumsum(random_walk) / Fs;
        end
        
        % 3. Sinusoidal FM component (simulate periodic oscillator instability)
        if contains(drift_params.type, 'sinusoidal') || contains(drift_params.type, 'combined')
            fm_signal = drift_params.fm_amplitude * sin(2*pi * drift_params.fm_frequency * t);
            % Integrate frequency offset to get phase
            phase_drift = phase_drift + 2*pi * cumsum(fm_signal) / Fs;
        end
        
        phase_drift = phase_drift(:);  % Ensure column vector
    end
    
    %% ========== Added: Generate symbol clock jitter function ==========
    function jittered_signal = apply_timing_jitter(signal, jitter_params, num_symbols, sps)
        % Apply symbol clock jitter to signal
        % Input:
        %   signal - Input signal
        %   jitter_params - Jitter parameters
        %   num_symbols - Number of symbols
        %   sps - Samples per symbol
        
        if ~jitter_params.enable
            jittered_signal = signal;
            return;
        end
        
        % Generate clock jitter for each symbol (unit: samples)
        if strcmp(jitter_params.type, 'gaussian')
            % Gaussian white noise jitter
            jitter_samples = randn(num_symbols, 1) * jitter_params.rms * sps;
        elseif strcmp(jitter_params.type, 'uniform')
            % Uniform distribution jitter
            jitter_samples = (rand(num_symbols, 1) - 0.5) * 2 * jitter_params.rms * sps * sqrt(3);
        else % 'colored'
            % Colored noise jitter (more realistic, adjacent symbol jitter is correlated)
            white_jitter = randn(num_symbols, 1);
            % First-order low-pass filter
            alpha_jitter = 0.3;
            jitter_samples = filter(alpha_jitter, [1, -(1-alpha_jitter)], white_jitter);
            jitter_samples = jitter_samples * jitter_params.rms * sps / std(jitter_samples);
        end
        
        % Use fractional delay filter to implement time-varying delay
        jittered_signal = zeros(size(signal));
        
        for sym_idx = 1:num_symbols
            % Current symbol's sampling point range
            start_idx = (sym_idx-1)*sps + 1;
            end_idx = min(sym_idx*sps, length(signal));
            
            if end_idx > length(signal)
                break;
            end
            
            % Extract current segment signal, add boundary check
            segment_start = max(1, start_idx-10);
            segment_end = min(length(signal), end_idx+10);
            
            % Ensure segment length is sufficient
            if segment_end >= segment_start
                segment = signal(segment_start:segment_end);
            else
                % If index is invalid, copy original signal segment
                jittered_signal(start_idx:end_idx) = signal(start_idx:end_idx);
                continue;
            end
            
            % Apply fractional delay (use Lagrange interpolation)
            delay_frac = jitter_samples(sym_idx);
            delay_int = floor(delay_frac);
            delay_frac_part = delay_frac - delay_int;
            
            % Third-order Lagrange interpolation
            if abs(delay_frac_part) > 0.001 && length(segment) >= 4
                segment_delayed = lagrange_interp(segment, delay_frac_part);
            else
                segment_delayed = segment;
            end
            
            % Integer delay
            if delay_int ~= 0
                if delay_int > 0
                    % Positive delay: pad zeros in front
                    if length(segment_delayed) > abs(delay_int)
                        segment_delayed = [zeros(abs(delay_int), 1); segment_delayed(1:end-abs(delay_int))];
                    else
                        % If delay is too large, just copy
                        segment_delayed = [zeros(abs(delay_int), 1); segment_delayed];
                    end
                else
                    % Negative delay: pad zeros at end
                    if length(segment_delayed) > abs(delay_int)
                        segment_delayed = [segment_delayed(abs(delay_int)+1:end); zeros(abs(delay_int), 1)];
                    else
                        % If delay is too large, just copy
                        segment_delayed = [segment_delayed; zeros(abs(delay_int), 1)];
                    end
                end
            end
            
            % Extract valid part, add boundary check
            valid_start = max(1, 11);
            valid_end = min(length(segment_delayed), valid_start + (end_idx - start_idx));
            
            if valid_end <= length(segment_delayed) && valid_start <= length(segment_delayed)
                % Ensure index is valid
                actual_length = min(end_idx - start_idx + 1, valid_end - valid_start + 1);
                if actual_length > 0
                    jittered_signal(start_idx:start_idx+actual_length-1) = segment_delayed(valid_start:valid_start+actual_length-1);
                end
            else
                % If interpolation fails, use original signal
                jittered_signal(start_idx:end_idx) = signal(start_idx:end_idx);
            end
        end
    end
    
    %% ========== Added: Lagrange interpolation function ==========
    function y_interp = lagrange_interp(y, delay_frac)
        % Third-order Lagrange fractional delay interpolation
        N = length(y);
        y_interp = zeros(N, 1);
        
        % Check if input length is sufficient
        if N < 4
            y_interp = y;
            return;
        end
        
        for n = 3:N-2
            % Use 4 points for third-order interpolation
            d = delay_frac;
            y_interp(n) = y(n-1) * (-d)*(d-1)*(d-2)/6 + ...
                          y(n)   * (d+1)*(d-1)*(d-2)/2 + ...
                          y(n+1) * (d+1)*d*(d-2)/(-2) + ...
                          y(n+2) * (d+1)*d*(d-1)/6;
        end
        
        % Boundary processing - only fill boundaries when original signal is long enough
        if N >= 4
            y_interp(1:2) = y(1:2);
            y_interp(N-1:N) = y(N-1:N);
        else
            y_interp = y;
        end
    end
    
    %% ========== Added: Generate amplitude variation function ==========
    function amplitude_envelope = generate_amplitude_variation(t, amp_params, file_seed)
        % Generate time-varying amplitude envelope
        % Input:
        %   t - Time vector
        %   amp_params - Amplitude variation parameters
        % Output:
        %   amplitude_envelope - Normalized amplitude envelope
        
        rng(file_seed * 8888);
        Fs = 1/mean(diff(t));
        N = length(t);
        
        amplitude_envelope = ones(N, 1);
        
        % 1. Fast random variation (simulate AGC, PA nonlinearity, etc.)
        if amp_params.variation_std > 0
            % Generate band-limited Gaussian noise
            white_noise = randn(N, 1);
            % Design low-pass filter to limit bandwidth
            [b_lp, a_lp] = butter(4, amp_params.variation_bandwidth/(Fs/2));
            fast_variation = filter(b_lp, a_lp, white_noise);
            fast_variation = fast_variation / std(fast_variation) * amp_params.variation_std;
            amplitude_envelope = amplitude_envelope + fast_variation;
        end
        
        % 2. Slow fading (simulate multipath, obstruction, etc.)
        if amp_params.fade_depth > 0
            % Sinusoidal fading
            slow_fade = amp_params.fade_depth * sin(2*pi * amp_params.fade_rate * t + rand()*2*pi);
            amplitude_envelope = amplitude_envelope .* (1 + slow_fade);
        end
        
        % Ensure amplitude is positive and reasonable
        amplitude_envelope = max(0.5, min(1.5, amplitude_envelope));
        amplitude_envelope = amplitude_envelope(:);  % Column vector
    end
    
    %% Prepare non-ideal characteristics parameter structure
    drift_params = struct('type', carrier_drift_type, ...
                          'rate', carrier_drift_rate, ...
                          'random_walk_std', carrier_drift_random_walk_std, ...
                          'fm_amplitude', carrier_fm_amplitude, ...
                          'fm_frequency', carrier_fm_frequency);
    
    jitter_params = struct('enable', enable_timing_jitter, ...
                           'rms', timing_jitter_rms, ...
                           'type', timing_jitter_type);
    
    amp_params = struct('variation_std', amplitude_variation_std, ...
                        'variation_bandwidth', amplitude_variation_bandwidth, ...
                        'fade_depth', amplitude_fade_depth, ...
                        'fade_rate', amplitude_fade_rate);
    
    %% ========================================================
    %% Generate complete signals file by file
    %% ========================================================
    
    for file_idx = 1:num_files
        fprintf('Generating file %d/%d\n', file_idx, num_files);
        
        %% Carrier frequency configuration (random frequency offset within ±700Hz for each file)
        fc_base = flo;  % Base carrier frequency matches the local oscillator
        
        switch num_sources
            case 2
                % Generate two symmetric frequency offsets within ±700Hz
                offset_mag = rand() * 700;  % Random value between 0 and 700Hz
                fc_offsets = [-offset_mag, offset_mag];
            case 3  
                % Generate one random value between -700 and 700Hz, plus 0Hz and symmetric value
                offset_mag = rand() * 700;  % Random value between 0 and 700Hz
                fc_offsets = [-offset_mag, 0, offset_mag];
            case 4
                % Generate two random frequency offsets, maintaining symmetry
                offset1 = rand() * 700;  % Random value between 0 and 700Hz
                offset2 = rand() * 700;  % Random value between 0 and 700Hz
                fc_offsets = [-offset1, -offset2, offset2, offset1];
        end
        
        % Calculate actual carrier frequencies
        if strict_tablea_8psk_a
            if strcmp(dataset_case, '8PSK-B')
                fc_offsets = [-250, +500];
            else
                fc_offsets = [-500, +500];
            end
        end
        fc_array = fc_base + fc_offsets;
        
        %% ========== Modified: Initial phase for each file is uniformly distributed in 0 to π ==========
        initial_phases = rand(num_sources, 1) * pi;  % Random phase for each source in 0 to π
        if strict_tablea_8psk_a
            if strcmp(dataset_case, '8PSK-B')
                initial_phases = [0; pi/5];
            else
                initial_phases = [0; pi/3];
            end
        end
        
        % Generate global time axis (continuous time starting from 0)
        t_global = (0:total_samples-1)' / Fs_rf;
        
        %% 1. Generate bit stream and modulation signals (multi-source)
        % Store signals for each source
        rf_signals = zeros(length(t_global), num_sources);
        ideal_bb_signals = zeros(length(t_global), num_sources);
        bit_data_all = cell(num_sources, 1);
        
        for src_idx = 1:num_sources
            %% Generate bit stream
            total_bits_src = total_symbols_by_source(src_idx) * bits_per_symbol;
            bit_data_all{src_idx} = randi([0, 1], total_bits_src, 1, 'uint8');
            
            %% Modulate to 8PSK symbols
            frame_symbol_labels = bi2de(reshape(bit_data_all{src_idx}, bits_per_symbol, [])', 'left-msb');

            % Use mapping table to convert to correct constellation index
            symbol_indices = gray_map_array(frame_symbol_labels + 1);
            s_complex = constellation(symbol_indices);
            
            %% Upsampling and pulse shaping
            Fs_sps_src = Fs_sps_by_source(src_idx);
            s_upsampled = upsample(s_complex, Fs_sps_src);
            s_shaped = conv(s_upsampled, filterCoeffs{src_idx}, 'same');
            
            %% ========== Key modification: Save ideal signal for target ==========
            s_shaped_ideal = s_shaped;
            
            %% Apply symbol clock jitter
            if enable_timing_jitter
                num_symbols_total = length(symbol_indices);
                s_shaped = apply_timing_jitter(s_shaped, jitter_params, num_symbols_total, Fs_sps_src);
                % Re-align length
                if length(s_shaped) > length(t_global)
                    s_shaped = s_shaped(1:length(t_global));
                elseif length(s_shaped) < length(t_global)
                    s_shaped = [s_shaped; zeros(length(t_global) - length(s_shaped), 1)];
                end
                s_shaped_ideal = s_shaped_ideal(1:length(s_shaped));
            else
                % If no jitter, ensure ideal signal length matches global time axis
                if length(s_shaped_ideal) > length(t_global)
                    s_shaped_ideal = s_shaped_ideal(1:length(t_global));
                elseif length(s_shaped_ideal) < length(t_global)
                    s_shaped_ideal = [s_shaped_ideal; zeros(length(t_global) - length(s_shaped_ideal), 1)];
                end
            end
            
            %% Add delay
            if delay_samples(src_idx) > 0
                s_shaped = [zeros(delay_samples(src_idx), 1); s_shaped(1:end-delay_samples(src_idx))];
                s_shaped_ideal = [zeros(delay_samples(src_idx), 1); s_shaped_ideal(1:end-delay_samples(src_idx))];
            end
            
            %% Ensure length consistency
            if length(s_shaped) > length(t_global)
                s_shaped = s_shaped(1:length(t_global));
            elseif length(s_shaped) < length(t_global)
                s_shaped = [s_shaped; zeros(length(t_global) - length(s_shaped), 1)];
            end
            
            %% Apply carrier frequency drift and amplitude variation
            if enable_carrier_drift
                phase_drift = generate_carrier_drift(t_global, fc_array(src_idx), drift_params, file_idx*100 + src_idx);
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + phase_drift + initial_phases(src_idx)));
            else
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end
            
            rf_signal = real(s_shaped .* carrier);
            
            %% Apply amplitude variation
            if enable_amplitude_variation
                amp_envelope = generate_amplitude_variation(t_global, amp_params, file_idx*200 + src_idx);
                rf_signal = rf_signal .* amp_envelope;
            end
            
            rf_signals(:, src_idx) = rf_signal;
            
            %% Generate ideal demodulation signal (label signal) - use ideal signal
            if enable_carrier_drift
                % If there is carrier drift, ideal signal should also have same drift
                phase_drift_ideal = generate_carrier_drift(t_global, fc_array(src_idx), drift_params, file_idx*100 + src_idx);
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + phase_drift_ideal + initial_phases(src_idx)));
            else
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end
            
            % Ensure ideal signal length matches time axis
            if length(s_shaped_ideal) ~= length(t_global)
                if length(s_shaped_ideal) > length(t_global)
                    s_shaped_ideal = s_shaped_ideal(1:length(t_global));
                else
                    s_shaped_ideal = [s_shaped_ideal; zeros(length(t_global) - length(s_shaped_ideal), 1)];
                end
            end
            
            rf_signal_ideal = real(s_shaped_ideal .* carrier_ideal);
            
            baseband_i = rf_signal_ideal .* cos(2*pi*flo*t_global);
            baseband_q = rf_signal_ideal .* (-sin(2*pi*flo*t_global));
            bb_i_filtered = conv(baseband_i, h_lpf_sources{src_idx}, 'same');
            bb_q_filtered = conv(baseband_q, h_lpf_sources{src_idx}, 'same');
            ideal_bb_signals(:, src_idx) = complex(bb_i_filtered, bb_q_filtered);
        end
        
        snr_emp_rf = NaN;
        snr_emp_bb = NaN;
        if strict_tablea_8psk_a
            % Strict Table(a) 8PSK-A: fixed params + configurable SNR domain.
            rf_combined_clean = sum(rf_signals, 2);

            % Downconvert clean mixture to baseband for reference.
            lo_i = cos(2*pi*flo*t_global);
            lo_q = sin(2*pi*flo*t_global);
            baseband_i_clean = rf_combined_clean .* lo_i;
            baseband_q_clean = rf_combined_clean .* (-lo_q);
            bb_i_clean = conv(baseband_i_clean, h_lpf_mixed, 'same');
            bb_q_clean = conv(baseband_q_clean, h_lpf_mixed, 'same');
            mixed_baseband_clean = complex(bb_i_clean, bb_q_clean);

            if strcmp(snr_domain, 'rf')
                rf_combined_noisy = awgn(rf_combined_clean, snr, 'measured');
                snr_emp_rf = empirical_snr_db_rf_pair(rf_combined_clean, rf_combined_noisy);

                baseband_i = rf_combined_noisy .* lo_i;
                baseband_q = rf_combined_noisy .* (-lo_q);
                bb_i_filtered = conv(baseband_i, h_lpf_mixed, 'same');
                bb_q_filtered = conv(baseband_q, h_lpf_mixed, 'same');
                mixed_baseband = complex(bb_i_filtered, bb_q_filtered);
            else
                mixed_baseband = awgn(mixed_baseband_clean, snr, 'measured');
            end

            snr_emp_bb = empirical_snr_db_bb(mixed_baseband_clean, mixed_baseband);
        else
            %% Mix signals and add noise
            rf_combined = sum(rf_signals, 2);
            rf_combined_noisy = awgn(rf_combined, snr, 'measured');

            %% Down-convert to baseband
            % Generate LO signal
            lo_i = cos(2*pi*flo*t_global);
            lo_q = sin(2*pi*flo*t_global);

            % Mix signal down-convert
            baseband_i = rf_combined_noisy .* lo_i;
            baseband_q = rf_combined_noisy .* (-lo_q);
            bb_i_filtered = conv(baseband_i, h_lpf_mixed, 'same');
            bb_q_filtered = conv(baseband_q, h_lpf_mixed, 'same');
            mixed_baseband = complex(bb_i_filtered, bb_q_filtered);
        end
        
        %% ========================================================
        %% File-level normalization
        %% ========================================================
        
        if strict_tablea_8psk_a
            % Shared normalization scale for mixture and targets to preserve SNR.
            scale = sqrt(mean(abs(mixed_baseband).^2));
            if isempty(scale) || scale < 1e-12
                scale = 1;
            end
            mixed_baseband = mixed_baseband ./ scale;
            ideal_bb_signals = ideal_bb_signals ./ scale;
        else
            % Calculate file-level maximum amplitude
            file_max_mixed = max(abs(mixed_baseband));
            file_max_ideal = max(max(abs(ideal_bb_signals)));

            % Apply file-level normalization
            mixed_baseband = mixed_baseband / file_max_mixed;
            ideal_bb_signals = ideal_bb_signals / file_max_ideal;
        end
        
        %% Split long signal into frames and save
        %% ========================================================
        
        % Reshape to frame structure
        mixed_frames = single(zeros(samples_per_file, frame_length, 2));
        ideal_frames = single(zeros(samples_per_file, frame_length, 2*num_sources)); % 2 channels (I,Q) per source
        
        % Frame processing
        for frame_idx = 1:samples_per_file
            % Calculate current frame position
            sample_start = (frame_idx-1)*frame_length + 1;
            sample_end = frame_idx * frame_length;
            if sample_end > length(mixed_baseband)
                break;
            end
            
            %% Mixed signal frame (directly use file normalized values)
            frame_data = mixed_baseband(sample_start:sample_end);
            mixed_frames(frame_idx, :, 1) = real(frame_data);
            mixed_frames(frame_idx, :, 2) = imag(frame_data);
            
            %% Ideal signal frame (directly use file normalized values)
            for src_idx = 1:num_sources
                frame_ideal = ideal_bb_signals(sample_start:sample_end, src_idx);
                ideal_frames(frame_idx, :, 2*src_idx-1) = real(frame_ideal);  % I channel
                ideal_frames(frame_idx, :, 2*src_idx) = imag(frame_ideal);    % Q channel
            end
        end
        
        %% Save data (project-relative by default; matches IQUMamba1D dataloader patterns)
        if nargin < 5 || isempty(output_root)
            script_dir = fileparts(mfilename('fullpath'));
            project_dir = fullfile(script_dir, '..');
            if strict_tablea_8psk_a
                output_root = fullfile(project_dir, 'data', 'synthetic', dataset_name);
            else
                output_root = fullfile(project_dir, 'data', 'synthetic', '8PSK_M');
            end
        end
        mix_dir = fullfile(output_root, 'mixture');
        tgt_dir = fullfile(output_root, 'target');
        bits_dir = fullfile(output_root, 'bits');
        if ~exist(mix_dir, 'dir'); mkdir(mix_dir); end %#ok<SEPEX>
        if ~exist(tgt_dir, 'dir'); mkdir(tgt_dir); end %#ok<SEPEX>
        if ~exist(bits_dir, 'dir'); mkdir(bits_dir); end %#ok<SEPEX>

        save_path_mixed = fullfile(mix_dir, sprintf('%dSource_%s_Dataset_mixed_%d_SNR=%ddB.mat', ...
            num_sources, dataset_name, file_idx, snr));
        save_path_target = fullfile(tgt_dir, sprintf('%dSource_%s_Dataset_target_%d_SNR=%ddB.mat', ...
            num_sources, dataset_name, file_idx, snr));

        rrc_alpha = alpha; %#ok<NASGU>
        rrc_span = span; %#ok<NASGU>
        save(save_path_mixed, 'mixed_frames', 'frame_length', 'valid_frame_length', 'symbols_per_frame', ...
            'Fs_sps', 'bits_per_symbol', 'bits_per_frame', 'rrc_alpha', 'rrc_span', '-v7.3');
        save(save_path_target, 'ideal_frames', 'frame_length', 'valid_frame_length', 'symbols_per_frame', ...
            'Fs_sps', 'bits_per_symbol', 'bits_per_frame', 'rrc_alpha', 'rrc_span', '-v7.3');

        % Save bit data (each source saved separately)
        for src_idx = 1:num_sources
            save_path_bit = fullfile(bits_dir, sprintf('%dSource_%s_BitData_%d_SNR=%ddB_Source%d.mat', ...
                num_sources, dataset_name, file_idx, snr, src_idx));
            file_bits = bit_data_all{src_idx}; %#ok<NASGU>
            save(save_path_bit, 'file_bits', 'bits_per_frame', 'bits_per_symbol', ...
                'symbols_per_frame', 'Fs_sps', 'rrc_alpha', 'rrc_span', '-v7.3');
        end

        if strict_tablea_8psk_a && file_idx == 1
            fprintf('  [Empirical SNR] RF=%.2f dB, BB=%.2f dB (target %d dB, domain=%s)\n', ...
                snr_emp_rf, snr_emp_bb, snr, snr_domain);
        end
        
        fprintf('File %d/%d saved (%d sources)\n', file_idx, num_files, num_sources);
        fprintf('  Frequency offset: ');
        for i = 1:num_sources
            fprintf('%.1f Hz ', fc_offsets(i));
        end
        fprintf('\n');
        fprintf('  Initial phase: ');
        for i = 1:num_sources
            fprintf('%.3f rad ', initial_phases(i));
        end
        fprintf('\n');
    end
    
    %% Display configuration information
    fprintf('\n=== Dataset generation completed ===\n');
    fprintf('Number of sources: %d\n', num_sources);
    fprintf('SNR: %d dB\n', snr);
    if strict_tablea_8psk_a
        if strcmp(dataset_case, '8PSK-B')
            fprintf('Frequency offset: [-250, +500] Hz\n');
            fprintf('Initial phase: [0, pi/5] rad\n');
            fprintf('Time delay: [0, 0.3Tb]\n');
            fprintf('Symbol rates: [2.5, 5.0] MHz\n');
            fprintf('Sampling rate: 50 MHz\n');
        else
            fprintf('Frequency offset: [-500, +500] Hz\n');
            fprintf('Initial phase: [0, pi/3] rad\n');
            fprintf('Time delay: [0, 0.05Tb]\n');
        end
    else
        fprintf('Frequency offset range: ±700 Hz\n');
    end
    fprintf('Number of files: %d\n', num_files);
    fprintf('Symbols per frame: %d\n', symbols_per_frame);
    fprintf('Bits per frame: %d\n', bits_per_frame);
    
    disp('All files generated');
end

function snr_db = empirical_snr_db_from_frames(mixed_frames, ideal_frames, num_sources)
    % mixed_frames: [B, L, 2], ideal_frames: [B, L, 2*num_sources]
    mf = double(mixed_frames);
    tf = double(ideal_frames);

    % to (B,2,L) and (B,2K,L)
    mix = permute(mf, [1, 3, 2]);
    tgt = permute(tf, [1, 3, 2]);

    sum_s = zeros(size(mix));
    for k = 1:num_sources
        ch = (2*k-1):(2*k);
        sum_s = sum_s + tgt(:, ch, :);
    end

    noise = mix - sum_s;

    p_sig = mean(sum(sum_s.^2, 2), 'all');
    p_noi = mean(sum(noise.^2, 2), 'all') + 1e-12;

    snr_db = 10 * log10(p_sig / p_noi);
end

function snr_domain = normalize_snr_domain(snr_domain)
    if isstring(snr_domain); snr_domain = char(snr_domain); end
    if ~ischar(snr_domain)
        error('snr_domain must be ''rf'' or ''bb''.');
    end
    snr_domain = lower(strtrim(snr_domain));
    if ~ismember(snr_domain, {'rf','bb'})
        error('snr_domain must be ''rf'' or ''bb''.');
    end
end

function dataset_name = normalize_tablea_8psk_dataset(dataset_name)
    if isstring(dataset_name); dataset_name = char(dataset_name); end
    if ~ischar(dataset_name)
        error('dataset_name must be ''8PSK-A'' or ''8PSK-B''.');
    end
    dataset_name = upper(strtrim(dataset_name));
    if ~ismember(dataset_name, {'8PSK-A', '8PSK-B'})
        error('Supported strict Table(a) 8PSK datasets are ''8PSK-A'' and ''8PSK-B''.');
    end
end

function snr_emp_rf = empirical_snr_db_rf_pair(rf_clean, rf_noisy)
    noise = rf_noisy - rf_clean;
    p_sig = mean(rf_clean.^2);
    p_noi = mean(noise.^2) + 1e-12;
    snr_emp_rf = 10 * log10(p_sig / p_noi);
end

function snr_emp_bb = empirical_snr_db_bb(bb_clean, bb_noisy)
    noise = bb_noisy - bb_clean;
    p_sig = mean(abs(bb_clean).^2);
    p_noi = mean(abs(noise).^2) + 1e-12;
    snr_emp_bb = 10 * log10(p_sig / p_noi);
end
