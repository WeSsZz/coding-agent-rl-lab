# route-v2 task split (provisional)

- **train**: 13 instances, audited test command on 13, repos {'conan-io/conan': 4, 'getmoto/moto': 7, 'pydantic/pydantic': 2}
    - `conan-io__conan-13230` conan-io/conan@2.0 f2p=1 p2p=34
    - `conan-io__conan-13450` conan-io/conan@2.0 f2p=1 p2p=9
    - `conan-io__conan-14397` conan-io/conan@2.0 f2p=2 p2p=2
    - `conan-io__conan-15121` conan-io/conan@2.0 f2p=1 p2p=3
    - `getmoto__moto-7317` getmoto/moto@5.0 f2p=1 p2p=52
    - `getmoto__moto-7324` getmoto/moto@5.0 f2p=2 p2p=40
    - `getmoto__moto-7335` getmoto/moto@5.0 f2p=2 p2p=20
    - `getmoto__moto-7336` getmoto/moto@5.0 f2p=5 p2p=16
    - `getmoto__moto-7347` getmoto/moto@5.0 f2p=2 p2p=3
    - `getmoto__moto-7348` getmoto/moto@5.0 f2p=1 p2p=40
    - `getmoto__moto-7584` getmoto/moto@5.0 f2p=1 p2p=19
    - `pydantic__pydantic-9004` pydantic/pydantic@2.7 f2p=1 p2p=60
    - `pydantic__pydantic-9214` pydantic/pydantic@2.7 f2p=1 p2p=54
- **development**: 10 instances, audited test command on 10, repos {'getmoto/moto': 10}
    - `getmoto__moto-7365` getmoto/moto@5.0 f2p=1 p2p=1
    - `getmoto__moto-7385` getmoto/moto@5.0 f2p=2 p2p=0
    - `getmoto__moto-7393` getmoto/moto@5.0 f2p=1 p2p=0
    - `getmoto__moto-7446` getmoto/moto@5.0 f2p=2 p2p=5
    - `getmoto__moto-7509` getmoto/moto@5.0 f2p=1 p2p=2
    - `getmoto__moto-7514` getmoto/moto@5.0 f2p=3 p2p=7
    - `getmoto__moto-7537` getmoto/moto@5.0 f2p=1 p2p=0
    - `getmoto__moto-7607` getmoto/moto@5.0 f2p=1 p2p=2
    - `getmoto__moto-7608` getmoto/moto@5.0 f2p=2 p2p=3
    - `getmoto__moto-7646` getmoto/moto@5.0 f2p=1 p2p=2
- **final-test**: 8 instances, audited test command on 8, repos {'conan-io/conan': 5, 'getmoto/moto': 2, 'pydantic/pydantic': 1}
    - `conan-io__conan-13721` conan-io/conan@2.0 f2p=1 p2p=6
    - `conan-io__conan-14051` conan-io/conan@2.0 f2p=1 p2p=15
    - `conan-io__conan-14362` conan-io/conan@2.0 f2p=2 p2p=32
    - `conan-io__conan-14616` conan-io/conan@2.0 f2p=2 p2p=34
    - `conan-io__conan-15007` conan-io/conan@2.0 f2p=1 p2p=6
    - `getmoto__moto-7371` getmoto/moto@5.0 f2p=2 p2p=55
    - `getmoto__moto-7524` getmoto/moto@5.0 f2p=1 p2p=18
    - `pydantic__pydantic-9170` pydantic/pydantic@2.7 f2p=1 p2p=17

pool: {'pool_instances': 63, 'pool_units': 60, 'pool_near_duplicate_groups': 3, 'pool_instances_in_a_duplicate_group': 6, 'unplaced_units': 30, 'development_from_route_v1': 10, 'cost_filter_dropped': {'pass_to_pass_over_cap': 25, 'fail_to_pass_over_cap': 4}, 'pool_instances_after_cost_filter': 63}
