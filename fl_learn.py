
small_env = '--min_length=12 --max_length=12 --num_layers=1 --hidden_dim=32 --num_cpus=10 --env_profile="small_default" --num_epoch=10000 '
big_env = '--min_length=20 --max_length=20 --num_layers=3 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --num_epoch=20000 '

large_env_base = '--num_layers=3 --hidden_dim=64 --num_cpus=20 --env_profile="large_default" --num_epoch=20000 '
small_env_base = '--num_layers=1 --hidden_dim=32 --num_cpus=10 --env_profile="small_default" --num_epoch=10000 '


base = 'python main.py --learn_mode=true --save_model=true --num_runs=5 --weight_map="all" '

trend_mode = [
    '"reward"',
    '"penalty"',
]

# trend_coef = ['0.3', '0.4', '0.5']
# iters = ['500', '1000', '1500']
trend_coef = ['0.3']
iters = ['1500']
s_env_seqs = ['12', '14', '16', '18', '20']
l_env_seqs = ['22', '24', '26', '28', '30']

cmd = ''

# with open('fl_script2.txt', 'w') as f:
#     for env in [small_env, big_env]:
#         for it in iters:
#             for tm in trend_mode:
#                 for tc in trend_coef:
#                     save_path = 's_{}_{}_{}_'.format(tc, tm.strip('"')[:2], it)
#                     if env == small_env:
#                         path_list = list(save_path)
#                         path_list[0] = 'g'
#                         save_path = ''.join(path_list)
#                     cmd = base + env + \
#                           "--trend_mode={} ".format(tm) + \
#                           "--trend_coef={} ".format(tc) + \
#                           "--cycle_iter={} ".format(it) + \
#                           "--save_to=save/{}".format(save_path)
#                     f.write(cmd + '\n')
#
#             # average
#             tm = 'ave'
#             tc = '0.5'
#             save_path = 's_{}_{}_'.format(tm, it)
#             if env == small_env:
#                 path_list = list(save_path)
#                 path_list[0] = 'g'
#                 save_path = ''.join(path_list)
#             cmd = base + env + \
#                   "--trend_mode={} ".format(tm) + \
#                   "--trend_coef={} ".format(tc) + \
#                   "--cycle_iter={} ".format(it) + \
#                   "--save_to=save/{}".format(save_path)
#             f.write(cmd + '\n')
#             f.write('\n')

with open('fl_script.txt', 'w') as f:
    for env_seq in l_env_seqs:
        env = "--min_length={} --max_length={} ".format(env_seq, env_seq) + large_env_base
        for it in iters:
            for tm in trend_mode:
                for tc in trend_coef:
                    save_path = 's_{}_{}_{}_{}_'.format(env_seq, tc, tm.strip('"')[:2], it)
                    if int(env_seq) >= 22:
                        path_list = list(save_path)
                        path_list[0] = 'l'
                        save_path = ''.join(path_list)
                    cmd = base + env + \
                          "--trend_mode={} ".format(tm) + \
                          "--trend_coef={} ".format(tc) + \
                          "--cycle_iter={} ".format(it) + \
                          "--save_to=save/{}".format(save_path)
                    f.write(cmd + '\n')

            # average
            tm = 'ave'
            tc = '0.5'
            save_path = 's_{}_{}_{}_'.format(env_seq,tm, it)
            if int(env_seq) >= 22:
                path_list = list(save_path)
                path_list[0] = 'l'
                save_path = ''.join(path_list)
            cmd = base + env + \
                  "--trend_mode={} ".format(tm) + \
                  "--trend_coef={} ".format(tc) + \
                  "--cycle_iter={} ".format(it) + \
                  "--save_to=save/{}".format(save_path)
            f.write(cmd + '\n')
            f.write('\n')
