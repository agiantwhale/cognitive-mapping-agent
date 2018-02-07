def write_to_file(prefix, map_nums):
    with open('{}_maps.txt'.format(prefix), 'w') as f:
        f.write(','.join('training-09x09-{:04d}'.format(map_num)
                         for map_num in map_nums))


def main():
    static_maps = [127, 169, 246, 336, 445, 589, 691, 828, 844, 956]

    write_to_file('static', static_maps)
    write_to_file('1', [1])
    write_to_file('10', range(1, 11))
    write_to_file('100', range(1, 101))
    write_to_file('1000', range(1, 1001))


if __name__ == '__main__':
    main()
