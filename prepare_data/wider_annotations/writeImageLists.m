function writeImageLists(image_set)

f=load(sprintf('wider_face_%s.mat', image_set));
fid = fopen(sprintf('%s.txt', image_set), 'a');
for i = 1 : length(f.event_list)
    for j = 1 : length(f.file_list{i})
        folder_name = f.event_list{i};
        file_name = f.file_list{i}{j};
        face_bboxes = f.face_bbx_list{i}{j};
        fprintf(fid, 'G:/mtcnn/data/WIDER_train/images/%s/%s', folder_name, file_name);
        fprintf(fid, '\r\n');
    end
end            
fclose(fid);        
        
