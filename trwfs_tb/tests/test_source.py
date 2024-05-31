from pylablib.devices import LighthousePhotonics

laser = LighthousePhotonics.SproutG("COM7")

laser.set_output_power(40)


laser.set_output_power(0)

laser.close()