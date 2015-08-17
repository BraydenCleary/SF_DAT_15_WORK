require 'json'

zip_geo_points = {}
leafit_data = []
zip = 0
paths_to_zips = Dir["./sf_zips/*.txt"]
paths_to_zips.each do |path|
  File.foreach(path).with_index do |line, line_num|
    if line_num == 0
      zip = line.match(/\d{5}/)[0].to_s
      zip_geo_points[zip] = []
    else
      zip_geo_points[zip] << line.strip.split(',').map(&:to_f).reverse
    end
  end
end

zip_geo_points.each_with_index do |(zip, coords), index|
  leafit_data << {
    type: "Feature",
    id: index.to_s,
    properties: {name: zip.to_s},
    geometry: {type: "Polygon", coordinates: [coords]}
  }
end

File.open('zip_data_compiled.rb', 'w') { |file| file.write(JSON.generate(leafit_data)) }
