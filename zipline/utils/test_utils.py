import zmq
import zipline.protocol as zp

def drain_zipline(test, zipline):
    assert test.ctx, "method expects a valid zmq context"
    assert test.zipline_test_config, "method expects a valid test config"
    assert isinstance(test.zipline_test_config, dict)
    assert test.zipline_test_config['results_socket'], \
            "need to specify a socket address for logs/perf/risk"
    test.receiver = test.ctx.socket(zmq.PULL)
    test.receiver.bind(test.zipline_test_config['results_socket'])

    output = []
    transaction_count  = 0
    while True:
        msg = test.receiver.recv()
        if msg == str(zp.CONTROL_PROTOCOL.DONE):
            break
        else:
            update = zp.BT_UPDATE_UNFRAME(msg)
            output.append(update)
            if update['prefix'] == 'PERF':
                transaction_count += \
                    len(update['payload']['daily_perf']['transactions'])

    del test.receiver

    # some processes will exit after the message stream is
    # finished. We block here to avoid collisions with subsequent
    # ziplines.
    for process in zipline.sim.subprocesses:
        process.join()

    return output, transaction_count


def assert_single_position(test, zipline):
    output, transaction_count = drain_zipline(test, zipline)

    test.assertTrue(zipline.sim.ready())
    test.assertFalse(zipline.sim.exception)

    test.assertEqual(
        test.zipline_test_config['order_count'],
        transaction_count
    )

    # the final message is the risk report, the second to
    # last is the final day's results. Positions is a list of
    # dicts.
    closing_positions = output[-2]['payload']['daily_perf']['positions']

    test.assertEqual(
        len(closing_positions),
        1,
        "Portfolio should have one position."
    )

    sid = test.zipline_test_config['sid']
    test.assertEqual(
        closing_positions[0]['sid'],
        sid,
        "Portfolio should have one position in " + str(sid)
    )
